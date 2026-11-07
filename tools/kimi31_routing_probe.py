"""Compare K31 latent projection and routing on real checkpoint weights.

The measured graphs include projections, column extraction, transport packing,
gathers, reduction and expert selection. They omit expert execution and are
component diagnostics, not native model throughput measurements.
"""

from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import tempfile


def worker(rank, checkpoint, destination, rendezvous):
    from ignition.recipes.kimi31.recipe import RECIPE
    os.environ.update(RECIPE.b12x_environment)
    import torch
    import torch.distributed as dist
    from ignition.components.collectives import PreparedChunkedAllGather, PreparedChunkedAllReduce
    from ignition.components.linear import UnquantizedProjection
    from ignition.components.moe import PreparedRouter, RouterScratch
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime, create_head_exchange,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.routing import PreparedLatentShardGather
    from ignition.recipes.kimi3.weights import KimiWeightLoader
    from b12x.preparation._measurement import _l2_flush_fn

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=16,
                            timeout=timedelta(minutes=15))
    shard = TensorParallelShard(rank, 16)
    prefix = "language_model.model.layers.1.block_sparse_moe"
    manifest = inspect_indexed_checkpoint(Path(checkpoint))
    with KimiWeightLoader(manifest, device, shard=shard,
            prefixes=(prefix + ".routed_expert_down_proj", prefix + ".gate")) as loader:
        full_down = loader.linear(prefix + ".routed_expert_down_proj")
        full_router = loader.linear(prefix + ".gate")
        correction = loader.load_tensor(prefix + ".gate.e_score_correction_bias")
    h0, h1 = shard.partition(7168)
    d0, d1 = shard.partition(3584)
    r0, r1 = shard.partition(896, alignment=8)
    column_weight = full_down[:, h0:h1].contiguous()
    row_weight = full_down[d0:d1].contiguous()
    router_weight = full_router[r0:r1].contiguous()
    coordinator = CollectivePreparation(dist.group.WORLD)
    reduction = create_all_reduce_runtime(dist.group.WORLD, device,
        collective_tuning="static", max_tokens=1024, hidden_size=7168)
    router_runtime = create_all_gather_runtime(dist.group.WORLD, device, max_rows=1024, columns=2048)
    combined_runtime = create_head_exchange(dist.group.WORLD, device, max_rows=1024, heads=16, head_dim=336)
    session = RecordedPreparationSession(device=device, checkpoint=str(checkpoint),
        workload_schema="kimi31-latent-routing", numerical_policy="default", coordinator=coordinator,
        compile_workers=16)
    buffers = {}
    cases, graphs, calls, measurements = [], [], [], []

    class Routing:
        def __init__(self, source, row_sharded, name):
            rows = source.shape[0]
            self.source, self.row_sharded = source, row_sharded
            self.columns = torch.arange(h0, h1, device=device)
            self.local_logits = torch.empty((rows, 56), dtype=torch.float32, device=device)
            self.logits = torch.empty((rows, 896), dtype=torch.float32, device=device)
            self.latent = torch.empty((rows, 3584), dtype=torch.bfloat16, device=device)
            self.projection = UnquantizedProjection(source, router_weight, self.local_logits,
                session=session, name=name + ".router")
            self.routes = RouterScratch.allocate(rows, device, experts=896, top_k=16)
            self.router = PreparedRouter(self.logits, self.routes, score_func="sigmoid",
                correction_bias=correction, renormalize=True, routed_scaling_factor=1.0,
                session=session, name=name + ".select")
            if row_sharded:
                self.local_down = torch.empty((rows, 224), dtype=torch.bfloat16, device=device)
                self.down = UnquantizedProjection(source, row_weight, self.local_down,
                    session=session, name=name + ".down")
                self.gather = PreparedLatentShardGather(self.local_down, self.local_logits,
                    self.latent, self.logits, runtime=combined_runtime, session=session,
                    coordinator=coordinator, name=name + ".gather", buffers=buffers)
            else:
                self.local_source = torch.empty((rows, 448), dtype=torch.bfloat16, device=device)
                self.partial = torch.empty_like(self.latent)
                self.down = UnquantizedProjection(self.local_source, column_weight, self.partial,
                    session=session, name=name + ".down")
                self.reduce = PreparedChunkedAllReduce(self.partial, self.latent,
                    runtime=reduction, session=session, coordinator=coordinator, name=name + ".reduce")
                self.send = torch.zeros((rows, 64), dtype=torch.float32, device=device)
                self.received = torch.empty((rows, 1024), dtype=torch.float32, device=device)
                self.gather = PreparedChunkedAllGather(self.send.view(torch.bfloat16),
                    self.received.view(torch.bfloat16), runtime=router_runtime, session=session,
                    coordinator=coordinator, name=name + ".gather")

        def __call__(self):
            self.projection(self.source)
            if self.row_sharded:
                self.down(self.source)
                self.gather()
                self.router(self.logits)
            else:
                self.send[:, :56].copy_(self.local_logits)
                self.gather(self.send.view(torch.bfloat16))
                self.logits.view(-1, 16, 56).copy_(self.received.view(-1, 16, 64)[:, :, :56])
                self.router(self.logits)
                self.local_source.copy_(self.source.index_select(1, self.columns))
                self.reduce(self.down(self.local_source))
            return self.latent

    try:
        for rows in (4096, 1024, 128):
            generator = torch.Generator(device=device).manual_seed(47311 + rows)
            source = torch.randn((rows, 7168), dtype=torch.bfloat16, device=device, generator=generator)
            variants = {name: Routing(source, name == "row", f"{name}-{rows}") for name in ("column", "row")}
            cases.append((rows, source, variants))
        session.tune(show_progress=False)
        flush = _l2_flush_fn(device, enabled=True)
        flush()
        for rows, source, variants in cases:
            modes = {}
            for mode in ("eager", "compiled"):
                if rank == 0:
                    print(f"Preparing routing fixture: rows={rows}, mode={mode}", flush=True)
                operations = {name: (operation if mode == "eager" else compile_forward(operation))
                              for name, operation in variants.items()}
                calls.extend(operations.values())
                for operation in operations.values():
                    operation()
                torch.cuda.synchronize()
                errors = []
                for seed in (8101, 8102, 8103):
                    generator = torch.Generator(device=device).manual_seed(seed + rows)
                    source.copy_(torch.randn(source.shape, dtype=source.dtype, device=device, generator=generator))
                    reference = torch.nn.functional.linear(source.float(), full_down.float()).bfloat16()
                    result = {}
                    for name, operation in operations.items():
                        variants[name].latent.fill_(float("nan"))
                        operation()
                        actual = variants[name].latent
                        difference = actual.float() - reference.float()
                        relative = float(difference.norm() / reference.float().norm())
                        assert relative < 0.01 and torch.isfinite(actual).all(), (name, rows, relative)
                        result[name] = dict(relative_l2=relative, max_absolute=float(difference.abs().max()))
                    assert torch.equal(variants["row"].logits, variants["column"].logits)
                    assert torch.equal(variants["row"].routes.ids, variants["column"].routes.ids)
                    assert torch.equal(variants["row"].routes.weights, variants["column"].routes.weights)
                    errors.append(result)
                timing, events, saved = {}, {}, {}
                for name, operation in operations.items():
                    pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                              torch.cuda.Event(enable_timing=True, external=True)) for _ in range(8)]
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with session.capture(), torch.cuda.graph(graph):
                        for start, stop in pairs:
                            flush()
                            start.record()
                            operation()
                            stop.record()
                    graphs.append(graph)
                    saved[name], events[name], timing[name] = graph, pairs, []
                for sequence in (("column", "row"), ("row", "column"), ("row", "column"), ("column", "row")):
                    for name in sequence:
                        dist.barrier()
                        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                        saved[name].replay()
                        torch.cuda.synchronize()
                        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                        sample = [start.elapsed_time(stop) for start, stop in events[name]]
                        ranks = [None] * 16
                        dist.all_gather_object(ranks, sample)
                        timing[name].extend(max(values) for values in zip(*ranks, strict=True))
                        actual = variants[name].latent
                        assert float((actual.float() - reference.float()).norm() / reference.float().norm()) < 0.01
                modes[mode] = dict(samples_ms=timing, median_ms={k: statistics.median(v) for k, v in timing.items()},
                                   errors=errors)
                if rank == 0:
                    print(json.dumps(dict(rows=rows, mode=mode, median_ms=modes[mode]["median_ms"], errors=errors)), flush=True)
            measurements.append(dict(rows=rows, modes=modes))
        if rank == 0:
            Path(destination).write_text(json.dumps(dict(checkpoint=checkpoint, world=16, layer=1,
                target_weight_dtype="bfloat16", target_activation_dtype="bfloat16", measurements=measurements), indent=2))
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        for graph in graphs:
            graph.reset()
        torch.cuda.synchronize()
        session.close()
        for runtime in (combined_runtime, router_runtime, reduction):
            runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="kimi-prefill-routing-") as directory:
        mp.spawn(worker, args=(arguments.checkpoint, arguments.output, Path(directory, "gloo").as_uri()),
                 nprocs=16, join=True)
