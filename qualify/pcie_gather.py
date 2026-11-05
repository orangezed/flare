#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check and time TP16 DCP gathers using package-owned CUDA IPC storage.

Run with --dry-run to inspect shapes and bounds without importing CUDA libraries.
Channels match Kimi's column and absorbed-query exchanges. Launch geometry is
explicit prepared metadata; weights are not loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import statistics


WORLD = 16


@dataclass(frozen=True)
class Geometry:
    name: str
    threads: int
    blocks: int
    active: int | None = None
    transport: str = "selected_barrier"


def geometries(include_selected=True, phased=False, ring=False):
    if ring:
        return (Geometry("ordinary_512x8", 512, 8),
                Geometry("phased_128x8", 128, 8, 8, "selected_phased_barrier"),
                Geometry("ring_128x8", 128, 8, 8, "selected_ring"))
    cases = [Geometry("ordinary_256x16", 256, 16),
             Geometry("ordinary_512x8", 512, 8), Geometry("gather_128x8", 128, 8)]
    cases += [Geometry(f"gather_{threads}x{blocks}", threads, blocks)
              for threads, blocks in ((64, 8), (128, 4), (128, 16), (128, 32), (256, 8), (512, 16))]
    if include_selected:
        cases += [Geometry(f"selected_128x32_active{active}", 128, 32, active)
                  for active in (8, 16, 32)]
    if phased:
        cases += [Geometry(f"phased_{threads}x8", threads, 8, 8, "selected_phased_barrier")
                  for threads in (512, 128, 256)]
        cases += [Geometry(f"push_{threads}x8", threads, 8, 8, "selected_phased_push")
                  for threads in (512, 128, 256)]
    return tuple(cases)


def validation_rows(args):
    if args.smoke:
        return [args.rows]
    return sorted({1, min(3, args.rows), max(1, args.rows - 1), args.rows}) if args.phased or args.epoch_check else [args.rows]


def payload_specs(args):
    families = [
        dict(name="embedding_context", dtype="bfloat16", local_heads=7, head_dim=64,
             max_rows=1024, capacity_rows=1024, rows=[1, 3, 8, 32, 128, 256, 512, 1024]),
        dict(name="latent", dtype="bfloat16", local_heads=1, head_dim=136,
             max_rows=1024, capacity_rows=1024, rows=[1, 3, 8, 32, 128, 256, 512, 1024]),
        dict(name="router", dtype="bfloat16_byte_view", local_heads=2, head_dim=64,
             max_rows=1024, capacity_rows=1024, rows=[9, 16, 32, 128, 256, 512, 1024]),
        dict(name="vocabulary", dtype="bfloat16", local_heads=160, head_dim=64,
             max_rows=32, capacity_rows=32, rows=[1, 2, 4, 7, 8, 14, 16, 28, 32]),
        dict(name="markov", dtype="bfloat16_byte_view", local_heads=320, head_dim=64,
             max_rows=4, capacity_rows=4, rows=[1, 2, 4]),
        dict(name="candidates", dtype="bfloat16_byte_view", local_heads=128, head_dim=64,
             max_rows=args.rows, capacity_rows=args.rows, rows=validation_rows(args)),
        dict(name="absorbed_queries", dtype="float8_e4m3fn", local_heads=6, head_dim=576,
             max_rows=args.rows, capacity_rows=args.head_channel_rows, rows=validation_rows(args)),
    ]
    if args.payloads != "both":
        families = [item for item in families if item["name"] == args.payloads]
    elif not args.model_families:
        families = [item for item in families if item["name"] in ("candidates", "absorbed_queries")]
    for family in families:
        if args.smoke:
            family["rows"] = [family["max_rows"]]
        family["row_bytes"] = family["local_heads"] * family["head_dim"] * (1 if family["dtype"] == "float8_e4m3fn" else 2)
        family["input_bytes"] = family["max_rows"] * family["row_bytes"]
        family["output_bytes"] = family["input_bytes"] * WORLD
    return families


def specification(args):
    return {"world": WORLD, "gpus": args.gpus, "rows": args.rows,
            "head_channel_rows": args.head_channel_rows,
            "payloads": payload_specs(args),
            "cases": [case.__dict__ for case in geometries(not args.no_selected, args.phased, args.ring)],
            "cache_policies": ["warm", "cold"], "samples": args.samples,
            "eviction_max_bytes": args.eviction_max_mib << 20,
            "compile_workers_per_host": args.compile_workers,
            "smoke": args.smoke,
            "timing": "one captured gather per sample; maximum CUDA duration across ranks"}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _agree(group, message):
    import torch.distributed as dist

    errors = [None] * WORLD
    dist.all_gather_object(errors, message, group=group)
    failures = [(rank, error) for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise AssertionError(f"gather validation failed: {failures}")


def _special_bits(fixture, rank, world):
    import torch

    rows = fixture.source.shape[0]
    row_bytes = fixture.source.nbytes // rows
    # Signed zero, -1 IDs, infinities and a NaN remain opaque payload bits,
    # including the ignored padding at both ends of a row.
    patterns = torch.tensor([0, 0, 0, 128, 255, 255, 255, 255,
                             0, 0, 128, 255, 1, 0, 192, 127],
                            dtype=torch.uint8, device=fixture.source.device)
    width = min(32, row_bytes)
    for seed in (0, 2):
        expected = fixture.reference.expected[seed].view(rows, world, row_bytes)
        pattern = patterns.repeat(2)[:width]
        if seed == 2:
            pattern = pattern.flip(0)
        expected[:, :, :width].copy_(pattern)
        expected[:, :, -width:].copy_(pattern)
        fixture.reference.seeds[seed].view(torch.uint8).view(rows, row_bytes).copy_(expected[:, rank, :])


def _graph_geometry(graph, expected):
    import cuda.bindings.driver as cuda

    raw = cuda.CUgraph(graph.raw_cuda_graph())
    status, _, count = cuda.cuGraphGetNodes(raw)
    assert status == cuda.CUresult.CUDA_SUCCESS and count == 1
    status, nodes, _ = cuda.cuGraphGetNodes(raw, count)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, parameters = cuda.cuGraphKernelNodeGetParams(nodes[0])
    assert status == cuda.CUresult.CUDA_SUCCESS
    grid = [parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ]
    block = [parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ]
    assert grid == [expected["physical_blocks"], 1, 1]
    assert block == [expected["threads"], 1, 1]
    status, attribute = cuda.cuGraphKernelNodeGetAttribute(
        nodes[0], cuda.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
    assert status == cuda.CUresult.CUDA_SUCCESS
    cluster = [attribute.clusterDim.x, attribute.clusterDim.y, attribute.clusterDim.z]
    if expected["cluster_blocks"] == 8:
        assert cluster == [8, 1, 1]
    else:
        assert cluster in ([0, 0, 0], [1, 1, 1])
    return {"grid": grid, "block": block, "cluster": cluster}


def _epoch_stress(rank, group, channels, fixtures, plans, captured, directory, sequence):
    import ctypes
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie._dcp_a2a_cute import _GRAPH_ARRIVED_INDEX, _GRAPH_EPOCH_INDEX

    stream = torch.cuda.current_stream()
    def word(runtime, index, value=None):
        host = ctypes.c_uint32(0 if value is None else value)
        address = runtime._signal_ptrs[rank] + 4 * index
        runtime._ipc.cudaMemcpyAsync(
            ctypes.addressof(host) if value is None else address,
            address if value is None else ctypes.addressof(host), 4, stream.cuda_stream)
        stream.synchronize()
        return host.value

    chosen = [name for name in captured if name.endswith(("/ordinary_512x8", "/phased_512x8", "/phased_128x8",
                                                        "/push_512x8", "/push_128x8", "/selected_128x32_active8",
                                                        "/ring_128x8"))]
    if sequence == "ordinary":
        chosen = [name for name in chosen if name.endswith("/ordinary_512x8")]
    elif sequence == "selected":
        chosen = [name for name in chosen if not name.endswith("/ordinary_512x8")]
    schedule = chosen + chosen[::-1]
    history = [(name, step % 3, torch.empty_like(fixtures[name.split("/")[0]].out))
               for step, name in enumerate(schedule)]
    expected_epoch = {}
    for payload, runtime, _ in channels:
        assert word(runtime, _GRAPH_ARRIVED_INDEX) == 0
        signal = (ctypes.c_uint32 * (64 * WORLD + 2 * 64 * WORLD * 32))()
        for block in range(64):
            for peer in range(WORLD):
                signal[block * WORLD + peer] = 0xFFFFFFFD
                for slot in range(2):
                    signal[64 * WORLD + (slot * 64 + block) * WORLD * 32 + peer * 32] = 0xFFFFFFFC + slot
        signal[_GRAPH_EPOCH_INDEX], signal[_GRAPH_ARRIVED_INDEX] = 0xFFFFFFFD, 0
        runtime._ipc.cudaMemcpyAsync(runtime._signal_ptrs[rank], ctypes.addressof(signal),
                                     ctypes.sizeof(signal), stream.cuda_stream)
        stream.synchronize()
        for bank in (runtime._staging0_ptrs, runtime._staging1_ptrs):
            runtime._ipc.cudaMemset(bank[rank], 0xA5, runtime._slot_bytes)
        expected_epoch[payload] = (0xFFFFFFFD + sum(
            name.split("/")[0].split("-rows")[0] == payload for name in schedule)) & 0xFFFFFFFF
    torch.cuda.synchronize()
    dist.barrier(group=group)
    counters = ("allocation.all.allocated", "allocation.all.freed",
                "segment.all.allocated", "segment.all.freed")
    before = torch.cuda.memory_stats()
    for name, seed, saved in history:
        fixture = fixtures[name.split("/")[0]]
        fixture.produce(seed)
        fixture.out.view(torch.uint8).fill_(0xA5)
        captured[name].replay()
        saved.copy_(fixture.out)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    errors = [key for key in counters if before[key] != after[key]]
    for position, (name, seed, saved) in enumerate(history):
        fixture = fixtures[name.split("/")[0]]
        if not torch.equal(saved.view(torch.uint8).reshape(-1), fixture.reference.expected[seed]):
            errors.append(f"queued output differs: {name}, seed {seed}")
            torch.save({"case": name, "seed": seed, "output": saved.cpu(),
                        "expected": fixture.reference.expected[seed].cpu()},
                       directory / f"rank-{rank}-queued-{sequence}-{position}-failure.pt")
    for prefix, fixture in fixtures.items():
        visits = [(name, seed) for name, seed, _ in history if name.split("/")[0] == prefix]
        if not visits:
            continue
        if not torch.equal(fixture.source.view(torch.uint8),
                           fixture.reference.seeds[visits[-1][1]].view(torch.uint8)):
            errors.append(f"queued replay modified source: {prefix}")
        if not torch.all(fixture.guarded[:16] == 0xA5) or not torch.all(fixture.guarded[-16:] == 0xA5):
            errors.append(f"queued replay overwrote guard: {prefix}")
    epochs = {}
    phase_counters = {}
    for payload, runtime, _ in channels:
        epochs[payload] = word(runtime, _GRAPH_EPOCH_INDEX)
        if epochs[payload] != expected_epoch[payload] or word(runtime, _GRAPH_ARRIVED_INDEX) != 0:
            errors.append(f"epoch mismatch: {payload}, {epochs[payload]}, expected {expected_epoch[payload]}")
        local_counters = (ctypes.c_uint32 * (8 * WORLD))()
        runtime._ipc.cudaMemcpyAsync(ctypes.addressof(local_counters), runtime._signal_ptrs[rank],
                                     ctypes.sizeof(local_counters), stream.cuda_stream)
        stream.synchronize()
        peers = [None] * WORLD
        dist.all_gather_object(peers, list(local_counters), group=group)
        if any(peers[left][block * WORLD + right] != peers[right][block * WORLD + left]
               for block in range(8) for left in range(WORLD) for right in range(WORLD)):
            errors.append(f"per-CTA paired peer generations differ: {payload}")
        phase_counters[payload] = [list(local_counters)[begin:begin + WORLD]
                                   for begin in range(0, 8 * WORLD, WORLD)]
    # A corrupt selector cannot partially enter the eight-CTA cluster barrier.
    for name in chosen:
        if not any(kind in name for kind in ("/phased_", "/push_", "/ring_")):
            continue
        state, fixture = plans[name].prepared.state, fixtures[name.split("/")[0]]
        initial = word(state.runtime, _GRAPH_EPOCH_INDEX)
        state.selector.fill_(7)
        fixture.out.view(torch.uint8).fill_(0xA5)
        dist.barrier(group=group)
        captured[name].replay()
        torch.cuda.synchronize()
        if word(state.runtime, _GRAPH_EPOCH_INDEX) != initial or not torch.all(fixture.out.view(torch.uint8) == 0xA5):
            errors.append(f"invalid profile modified output or IPC state: {name}")
        state.selector.fill_(8)
    _json(directory / f"rank-{rank}-epoch-{sequence}.json",
          {"queued_replays": len(schedule), "initial_epoch": 0xFFFFFFFD, "sequence": sequence,
           "bank_origins": {payload: runtime._graph_base_slot for payload, runtime, _ in channels},
           "per_cta_peer_generations": phase_counters,
           "final_epochs": epochs, "expected_epochs": expected_epoch,
           "allocator_counters_stable": not any(key in errors for key in counters),
           "byte_exact": not errors, "errors": errors})
    _agree(group, errors or None)


def _rank(rank, args, rendezvous):
    import torch
    import torch.distributed as dist
    from b12x.comm import pcie
    from b12x.comm.pcie._gather_preparation import GatherInputs, selected_gather_profile
    from b12x.preparation import CollectiveRequirement, FrozenMapping, PreparedCall
    from ignition.preparation.collectives import (
        CollectivePreparation, create_all_gather_runtime, create_head_exchange,
    )
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    properties = torch.cuda.get_device_properties(device)
    eviction_bytes = max(2 * properties.L2_cache_size, 64 << 20)
    _agree(group, None if eviction_bytes <= args.eviction_max_mib << 20
           else f"L2 eviction needs {eviction_bytes} bytes, exceeding the configured bound")
    directory = Path(args.output)
    with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
        coordinator = CollectivePreparation(group)
        channels = []
        families = {item["name"]: item for item in payload_specs(args)}
        for family in families.values():
            payload, capacity = family["name"], family["capacity_rows"]
            if payload in ("latent", "absorbed_queries"):
                runtime = create_head_exchange(group, device, max_rows=capacity,
                    heads=family["local_heads"] * WORLD,
                    head_dim=512 if payload == "absorbed_queries" else family["head_dim"],
                    query_head_dim=family["head_dim"], ipc=storage.peer_storage)
            else:
                runtime = create_all_gather_runtime(group, device, max_rows=capacity,
                    columns=family["local_heads"] * family["head_dim"] * WORLD,
                    ipc=storage.peer_storage)
            channels.append((payload, runtime,
                torch.float8_e4m3fn if family["dtype"] == "float8_e4m3fn" else torch.bfloat16))
        plans, fixtures, metadata, captured, records = {}, {}, {}, {}, []
        with RecordedPreparationSession(device=device, checkpoint="synthetic-tp16-gather-bytes",
                workload_schema="tp16-column-gather-families",
                numerical_policy="byte-exact", coordinator=coordinator,
                compile_workers=args.compile_workers) as session:
            for payload, runtime, dtype in channels:
                for rows in families[payload]["rows"]:
                    prefix = payload if rows == families[payload]["max_rows"] else f"{payload}-rows{rows}"
                    source = torch.empty((rows, runtime.heads_per_rank, runtime.query_head_dim),
                                         dtype=dtype, device=device)
                    output = torch.empty((rows, runtime.total_heads, runtime.query_head_dim),
                                         dtype=dtype, device=device)
                    base = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_heads",
                                                   call={"local_input": source, "out": output})
                    # Query metadata contains the resolved dependency settings. Refuse
                    # an inherited override instead of racing mislabeled geometries.
                    _agree(group, None if (base.call["threads"], base.call["block_limit"]) == (256, 16)
                           and not base.setup["threads_override"] and not base.setup["blocks_override"]
                           else f"{payload}: inherited settings forced {dict(base.call)}")
                    from types import SimpleNamespace
                    fixture = GatherInputs(SimpleNamespace(query=base, runtime=runtime))
                    _special_bits(fixture, rank, WORLD)
                    fixture.guarded = torch.full((fixture.out.nbytes + 32,), 0xA5,
                                                  dtype=torch.uint8, device=device)
                    fixture.out = fixture.guarded[16:-16].view(dtype).view(fixture.out.shape)
                    fixtures[prefix] = fixture
                    del source, output
                    for geometry in geometries(not args.no_selected, args.phased, args.ring):
                        name = f"{prefix}/{geometry.name}"
                        if geometry.name.startswith("factory_default"):
                            query = base
                        else:
                            query = pcie.query_from_runtime(runtime, surface=base.surface,
                                call={"local_input": fixture.source, "out": fixture.out,
                                      "threads": geometry.threads, "block_limit": geometry.blocks})
                        if geometry.active is not None:
                            query = replace(query, call=FrozenMapping(dict(query.call,
                                gather_transport=geometry.transport, submode="barrier",
                                active_blocks=geometry.active)))
                        effective = runtime._resolve_launch_config(threads=query.call["threads"],
                                                                   block_limit=query.call["block_limit"])
                        _agree(group, None if effective == (geometry.threads, geometry.blocks)
                               else f"{name}: effective geometry {effective} differs from its label")
                        plan = pcie.plan(query, runtime=runtime)
                        plans[name] = plan
                        metadata[name] = {"shape": list(query.call["shape"]), "dtype": query.call["dtype"],
                            "measure_rows": families[payload]["max_rows"],
                            "query": dict(query.call), "setup": dict(query.setup),
                            "threads": effective[0], "block_limit": effective[1],
                            "physical_blocks": effective[1] if geometry.active is not None else min(effective[1],
                                (rows * runtime.total_heads + effective[0] // 32 - 1) // (effective[0] // 32)),
                            "active_blocks": geometry.active,
                            "cluster_blocks": 8 if geometry.transport.startswith("selected_phased") else 1,
                            "input_bytes": fixture.source.nbytes, "output_bytes": fixture.out.nbytes}
                        def prepare(state, fixture=fixture):
                            fixture.produce(0)
                            return PreparedCall(run=lambda: state.run(local_input=fixture.source, out=fixture.out),
                                                output=fixture.out, owners=(fixture,))
                        session.declare((plan.request(name=name, prepare_call=prepare,
                            collective=CollectiveRequirement(key=name, ranks=coordinator.ranks)),),
                            coordinator=coordinator)
            session.tune(show_progress=False)
            for name, plan in plans.items():
                fixture = fixtures[name.split("/")[0]]
                state = plan.prepared.state
                effective = state.runtime._resolve_launch_config(
                    threads=state.query.call["threads"], block_limit=state.query.call["block_limit"])
                assert effective == (metadata[name]["threads"], metadata[name]["block_limit"])
                if "active_blocks" in state.query.call:
                    expected_profile = selected_gather_profile(state.query.call["gather_transport"], effective[1], "barrier",
                                                               metadata[name]["active_blocks"])
                    assert state.selector.item() == expected_profile
                fixture.produce(0)
                state.run(local_input=fixture.source, out=fixture.out)
                if not state.runtime._device_slot_selection and state.runtime._next_slot != args.graph_base_slot:
                    state.run(local_input=fixture.source, out=fixture.out)
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with session.capture(), torch.cuda.graph(graph):
                    state.run(local_input=fixture.source, out=fixture.out)
                metadata[name]["captured_geometry"] = _graph_geometry(graph, metadata[name])
                metadata[name]["graph_base_slot"] = state.runtime._graph_base_slot
                assert state.runtime._graph_base_slot == args.graph_base_slot
                captured[name] = graph
            session.freeze()
            eviction = torch.ones(eviction_bytes // 4, dtype=torch.float32, device=device)
            eviction_sum = torch.empty((), dtype=torch.float32, device=device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            snapshot = {"rank": rank, "device": properties.name, "l2_bytes": properties.L2_cache_size,
                        "eviction_bytes": eviction_bytes, "allocated_bytes": torch.cuda.memory_allocated(device),
                        "reserved_bytes": torch.cuda.memory_reserved(device), "launches": metadata}
            _json(directory / f"rank-{rank}-launches.json", snapshot)
            names = tuple(captured)
            timed_names = tuple(name for name in names if metadata[name]["shape"][0] == metadata[name]["measure_rows"])
            counters = ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed")

            def replay(name, seed, cold=False, timed=False):
                fixture, graph = fixtures[name.split("/")[0]], captured[name]
                fixture.produce(seed)
                fixture.out.view(torch.uint8).fill_(0xA5)
                if cold:
                    torch.sum(eviction, dim=0, out=eviction_sum)
                elif timed:
                    graph.replay()
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                before = torch.cuda.memory_stats(device)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                after = torch.cuda.memory_stats(device)
                elapsed = start.elapsed_time(end) * 1000
                error = None
                if any(before[key] != after[key] for key in counters):
                    error = f"{name}: replay changed allocator counters"
                if not torch.equal(fixture.out.view(torch.uint8).reshape(-1), fixture.reference.expected[seed]):
                    torch.save({"source": fixture.source.cpu(), "output": fixture.out.cpu(),
                                "expected": fixture.reference.expected[seed].cpu()},
                               directory / f"rank-{rank}-failure.pt")
                    error = f"{name}: captured output differs from the exact rank-interleaved bytes"
                if not torch.all(fixture.guarded[:16] == 0xA5) or not torch.all(fixture.guarded[-16:] == 0xA5):
                    error = f"{name}: output guard was overwritten"
                _agree(group, error)
                durations = [None] * WORLD
                dist.all_gather_object(durations, elapsed, group=group)
                return durations

            # Vary contents and alternate launch geometries on the same IPC owner.
            for seed in range(3):
                for name in names[::(-1 if seed % 2 else 1)]:
                    replay(name, seed)
            if args.phased or args.epoch_check:
                for sequence in ("ordinary", "selected", "mixed"):
                    _epoch_stress(rank, group, channels, fixtures, plans, captured, directory, sequence)
            for sample in range(args.samples):
                ordered = timed_names[sample % len(timed_names):] + timed_names[:sample % len(timed_names)]
                if sample % 2:
                    ordered = ordered[::-1]
                for name in ordered:
                    for policy in (("warm", "cold") if sample % 2 else ("cold", "warm")):
                        durations = replay(name, sample % 3, cold=policy == "cold", timed=True)
                        row = {"case": name, "cache_policy": policy, "sample": sample,
                               "rank_us": durations, "max_rank_us": max(durations)}
                        if rank == 0:
                            records.append(row)
                            with (directory / "samples.jsonl").open("a") as output:
                                output.write(json.dumps(row) + "\n")
                            print(json.dumps(row), flush=True)
            if rank == 0:
                summary = []
                for name in timed_names:
                    for policy in ("warm", "cold"):
                        values = [row["max_rank_us"] for row in records
                                  if row["case"] == name and row["cache_policy"] == policy]
                        summary.append({"case": name, "cache_policy": policy,
                            "median_max_rank_us": statistics.median(values),
                            "minimum_max_rank_us": min(values), "maximum_max_rank_us": max(values)})
                _json(directory / "summary.json", summary)
            for graph in captured.values():
                graph.reset()
            captured.clear()
            del graph, eviction, eviction_sum, start, end
        for _, runtime, _ in reversed(channels):
            runtime.close()
        del state, plan, plans, fixture, fixtures, captured
    dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0-15")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--head-channel-rows", type=int, default=128)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--eviction-max-mib", type=int, default=256)
    parser.add_argument("--compile-workers", type=int, default=16)
    parser.add_argument("--no-selected", action="store_true")
    parser.add_argument("--phased", action="store_true")
    parser.add_argument("--ring", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-families", action="store_true")
    parser.add_argument("--payloads", choices=("both", "candidates", "absorbed_queries", "embedding_context", "latent", "router", "vocabulary", "markov"), default="both")
    parser.add_argument("--epoch-check", action="store_true")
    parser.add_argument("--graph-base-slot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.ring:
        args.phased = True
    devices = [int(value) for part in args.gpus.split(",") for value in (
        range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else (int(part),))]
    if len(devices) != WORLD or len(set(devices)) != WORLD or min(devices) < 0:
        parser.error("--gpus must select exactly sixteen distinct nonnegative GPU ordinals")
    if not 1 <= args.rows <= args.head_channel_rows or args.samples < 1 or args.eviction_max_mib < 64:
        parser.error("require positive row/sample capacities and at least 64 MiB for eviction")
    if args.compile_workers < 1:
        parser.error("--compile-workers must be positive")
    if args.dry_run:
        print(json.dumps(specification(args), indent=2))
        return 0
    if args.output.exists():
        parser.error("--output must be an unused directory")
    args.output.mkdir(parents=True)
    from types import SimpleNamespace
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    apply(Configuration.load(args.config), [SimpleNamespace(uuid=str(device)) for device in devices])
    import b12x
    import torch
    import torch.multiprocessing as mp
    if torch.cuda.device_count() != WORLD:
        raise RuntimeError("sixteen selected CUDA devices are required")
    spec = specification(args)
    for name, directory in (("flare_revision", Path(__file__).resolve().parents[1]),
                            ("flashinfer_revision", Path(b12x.__file__).resolve().parent)):
        spec[name] = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    _json(args.output / "run.json", spec)
    hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.current.graphics,clocks.current.memory", "--format=csv"],
                              text=True, capture_output=True, check=True)
    (args.output / "gpus-before.csv").write_text(hardware.stdout)
    mp.spawn(_rank, args=(args, (args.output.resolve() / "rendezvous").as_uri()), nprocs=WORLD, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
