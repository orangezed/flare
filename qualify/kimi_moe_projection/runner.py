"""Qualify compiled Kimi MoE projection overlap against serial execution on sixteen GPUs.

Loads one checkpoint layer and uses the same tuned plans for both schedules.
Three input seeds, forward/reverse execution, poisoned scratch and allocation
counters check native graph replay. Timings include the complete MoE layer and
its output reduction, but do not establish whole-model serving throughput."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace


def rank_main(rank, world, rendezvous, checkpoint, destination, rows, cache_root):
    import torch
    import torch.distributed as dist
    from b12x.preparation._measurement import _l2_flush_fn
    from ignition.components.collectives import PreparedChunkedAllReduce
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.decoder import DecoderStorage, PreparedLatentMoE
    from ignition.recipes.kimi3.routing import create_route_runtime
    from ignition.recipes.kimi3.weights import KimiWeightLoader

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=15))
    shard = TensorParallelShard(rank, world)
    with PackageStorage(rank=rank, world=world, device=rank) as storage:
        manifest = inspect_indexed_checkpoint(Path(checkpoint))
        with KimiWeightLoader(manifest, device, shard=shard,
                prefixes=("language_model.model.layers.1.",), online_mxfp8=True,
                online_mxfp8_scope="kda-input", peers=(dist.group.WORLD, storage.peer_storage)) as loader:
            layer = loader.load_layer(1)
            config = loader.config
            scale_scratch = loader.scale_scratch
        if rank == 0:
            print("Checkpoint layer 1 loaded on all context ranks", flush=True)
        coordinator = CollectivePreparation(dist.group.WORLD)
        runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=rows,
            hidden_size=config.hidden, ipc=storage.peer_storage, collective_tuning="static")
        routing = create_route_runtime(dist.group.WORLD, device, ipc=storage.peer_storage)
        generator = torch.Generator(device=device).manual_seed(7331)
        source = torch.randn((rows, config.hidden), device=device, dtype=torch.bfloat16, generator=generator)
        total = torch.empty_like(source)
        with RecordedPreparationSession(device=device, checkpoint=Path(checkpoint).name,
                workload_schema="kimi3", numerical_policy="mxfp8-kda-input", coordinator=coordinator,
                compile_workers=16, benchmark_storage=storage, collective_tuning="static") as session:
            arena, pool = ScratchArena(device, overlap=True), GraphBufferPool(device, chunk_bytes=2 << 20)
            session.register_scratch(arena)
            operation = PreparedLatentMoE(layer, source, config, shard, runtime=runtime,
                router_runtime=None, route_runtime=routing, coordinator=coordinator, session=session,
                name="layer1.moe", workspace=arena, buffers={},
                storage=DecoderStorage.allocate(config, shard, device), allocator=pool.cursor())
            reduction = PreparedChunkedAllReduce(operation.output, total, runtime=runtime,
                coordinator=coordinator, session=session, name="layer1.reduce")
            session.tune(show_progress=False)

            assert operation.shared.concurrent and operation.route_gather is not None
            assert operation.concurrent_down is not None
            names = ('serial', 'recipe')
            from .cases import MoEProjectionSchedule
            functions = {name: MoEProjectionSchedule(operation, source, reduction, name) for name in names}
            signatures = {name: GraphSignature('decode', rows, 1, ()) for name in names}
            correctness_graphs, compiled = capture_forwards(functions, signatures,
                session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=storage.graph_pool.id, coordinator=coordinator, compile_decoder=True,
                cache_root=Path(cache_root),
                warmup_storage=storage, retain_graphs=True,
                progress=(lambda value: print(value, flush=True)) if rank == 0 else None)

            flush = _l2_flush_fn(device, enabled=True)
            flush()
            timed_graphs, events = {}, {}
            for name in names:
                execute = compiled[name]
                pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                          torch.cuda.Event(enable_timing=True, external=True)) for _ in range(32)]
                timed = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(timed):
                    for begin, end in pairs:
                        flush()
                        begin.record()
                        execute()
                        end.record()
                timed_graphs[name], events[name] = timed, pairs
            session.freeze()

            samples = []
            for seed in (7331, 9107, 11413):
                generator.manual_seed(seed)
                source.copy_(torch.randn(source.shape, device=device, dtype=source.dtype, generator=generator))
                expected = None
                for name in (*names, *reversed(names)):
                    arena.tensor.fill_(255)
                    for tensor in scale_scratch:
                        tensor.fill_(255)
                    total.fill_(float("nan"))
                    dist.barrier()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    correctness_graphs[name].replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                    assert torch.isfinite(total).all()
                    actual = total.clone()
                    if expected is None:
                        expected = actual
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name in (*names, *reversed(names)):
                    dist.barrier()
                    timed_graphs[name].replay()
                    torch.cuda.synchronize()
                    durations = [begin.elapsed_time(end) * 1000 for begin, end in events[name]]
                    ranks = [None] * world
                    dist.all_gather_object(ranks, durations)
                    if rank == 0:
                        maximums = [max(values) for values in zip(*ranks)]
                        samples.append(dict(seed=seed, projection_schedule=name,
                            per_rank_us=ranks, maximum_rank_us=maximums,
                            discarded_startup_steps=2, measured_maximum_rank_us=maximums[2:]))
                        Path(destination, "samples.json").write_text(json.dumps(samples, indent=2))
                        print(f"seed={seed} projection_schedule={name}: "
                              f"{statistics.mean(maximums[2:]):.3f} us", flush=True)
            for graph in (*correctness_graphs.values(), *timed_graphs.values()):
                graph.reset()
            if rank == 0:
                print("PASSED exact matched-plan outputs and allocation-stable replay; component timing only", flush=True)
        dist.barrier()
        routing.close()
        runtime.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, choices=(1,2,4,8), default=1)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                        "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError(f"GPUs are occupied; no benchmark started: {occupied}")
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if len(uuids) != 16:
        raise RuntimeError("This comparison requires all sixteen assigned GPUs")
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.recipes.kimi31.recipe import RECIPE
    configuration = Configuration.load(args.config)
    apply(configuration,
          [SimpleNamespace(uuid=uuid.strip()) for uuid in uuids], RECIPE.b12x_environment.items())
    import torch.multiprocessing as mp
    args.output.mkdir(parents=True, exist_ok=False)
    mp.spawn(rank_main, args=(16, (args.output / "rendezvous").resolve().as_uri(),
                             args.checkpoint, str(args.output), args.rows,
                             str(Path(configuration.get("paths.kernel_cache").value).expanduser())),
             nprocs=16, join=True)


if __name__ == "__main__":
    main()
