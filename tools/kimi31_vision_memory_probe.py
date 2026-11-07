"""Measure compiled replicated BF16 MoonViT against its unchanged eager forward."""

import argparse
import json
import os
from pathlib import Path
import sys


from vision_feature_capture import Features


def fixture(height, width, image, device):
    import torch

    n = height * width
    generator = torch.Generator(device="cpu").manual_seed(411 + image)
    pixels = torch.randn((n, 588), generator=generator, dtype=torch.bfloat16)
    positions = torch.cartesian_prod(torch.arange(height), torch.arange(width))
    mapped = ((positions.float() + .5) * torch.tensor((64 / height, 64 / width)) - .5).clamp_min(0)
    base, fraction = mapped.long(), mapped - mapped.long()
    upper = (base + 1).clamp_max(63)
    indices = torch.stack((base[:, 0] * 64 + base[:, 1], base[:, 0] * 64 + upper[:, 1],
                          upper[:, 0] * 64 + base[:, 1], upper[:, 0] * 64 + upper[:, 1]), dim=1)
    fy, fx = fraction.unbind(1)
    factors = torch.stack(((1-fy)*(1-fx), (1-fy)*fx, fy*(1-fx), fy*fx), dim=1)
    merge = torch.arange(n).view(height//2, 2, width//2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    return tuple(value.to(device) for value in (pixels, positions, indices, factors, merge))


def restore(encoder, values):
    import torch

    pixels, positions, indices, factors, merge = values
    for destination, value in ((encoder.scratch.patches, pixels), (encoder.positions, positions),
            (encoder.interpolation_indices, indices), (encoder.interpolation_weights, factors),
            (encoder.merge_indices, merge)):
        destination.zero_()
        destination[:value.shape[0]].copy_(value)
    encoder.cu.copy_(torch.tensor([0, pixels.shape[0]], dtype=torch.int32, device=pixels.device))
    encoder.num_patches.fill_(pixels.shape[0])
    encoder.num_features.fill_(pixels.shape[0]//4)


def run(args):
    import torch
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.vision import PATCH_CAPACITIES, VisionInputs, VisionScratch, VisionWeights, PreparedVision
    from ignition.recipes.kimi3.vision_state import PreparedVisionGeneration
    from ignition.components.completion import CompletionOutputs
    from ignition.components.paged_batch import BatchInputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState

    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if torch.cuda.device_count() != 1:
        raise ValueError("expose one assigned GPU through CUDA_VISIBLE_DEVICES")
    ns = (json.loads(args.selection_cache.read_text())["identity"]["namespace"]
          if args.selection_cache is not None else dict(checkpoint=str(args.checkpoint.resolve()),
              workload_schema="kimi31-vision-memory", numerical_policy="default"))
    session = RecordedPreparationSession(device=device, checkpoint=ns["checkpoint"],
        workload_schema=ns["workload_schema"], numerical_policy=ns["numerical_policy"], compile_workers=8)
    weights = VisionWeights.load(inspect_indexed_checkpoint(args.checkpoint), device)
    arena = ScratchArena(device)
    session.register_scratch(arena)
    pool, inputs = GraphBufferPool(device, chunk_bytes=2 << 20), VisionInputs.allocate(device)
    output = None
    if args.publication:
        requests = DeviceRequestState.allocate(device, vocab_size=163840, capacity=1)
        batch = BatchInputs.allocate(4096, 1, device, capacity=1)
        state = VisionState(batch, PrefillVisionInputs.allocate(4096, device), requests,
            image_token=163605, features=16384, hidden=448, image_slots=1)
        completion = CompletionOutputs.allocate((2, 1, 24), device=device)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.ones(1, dtype=torch.int64, device=device)
    else:
        output = torch.empty((16384, 7168), dtype=torch.bfloat16, device=device)
    encoders, forwards, signatures, values, references = {}, {}, {}, {}, {}
    for n in reversed(PATCH_CAPACITIES):
        cursor = pool.cursor()
        scratch = VisionScratch.allocate(n, device, buffers=cursor)
        name = f"vision{n}"
        encoder = PreparedVision(weights, scratch, inputs, session=session, workspace=arena,
                                 name=name, buffers=cursor)
        encoders[name] = encoder
        forwards[name] = (PreparedVisionGeneration(encoder, state, completion, frame, sequence,
            feature_range=(0, 448), buffers=cursor) if args.publication else Features(encoder, output[:n//4]))
        signatures[name] = GraphSignature("vision", 1, n)
        side = int(n**.5)
        values[name] = fixture(side, side, 0, device)
    print("Prepare complete vision catalog", flush=True)
    session.tune(show_progress=False)
    def restore_inputs(name, fields):
        restore(encoders[name], fields)
        if args.publication:
            pixels, positions, indices, factors, merge = fields
            for destination, value in ((inputs.vision_pixel_values, pixels),
                    (inputs.vision_position_ids, positions), (inputs.vision_interpolation_indices, indices),
                    (inputs.vision_interpolation_weights, factors), (inputs.vision_merge_indices, merge)):
                destination.zero_()
                destination[:value.shape[0]].copy_(value)
            inputs.vision_cu_seqlens.copy_(torch.tensor([0, pixels.shape[0]], dtype=torch.int32, device=device))
            inputs.vision_num_patches.fill_(pixels.shape[0])
            inputs.vision_num_features.fill_(pixels.shape[0]//4)
            inputs.vision_slot.zero_()
            inputs.vision_generation.fill_(17)
            inputs.vision_feature_offset.zero_()
            requests.status.fill_(1)
            requests.generation.fill_(17)
            state.feature_slot.zero_()
        else:
            forwards[name].pixels.zero_()
            forwards[name].pixels[:fields[0].shape[0]].copy_(fields[0])
    def observed(name):
        return state.arena[0, :encoders[name].scratch.patches.shape[0]//4] if args.publication else forwards[name].output
    for name, encoder in encoders.items():
        restore(encoder, values[name])
        reference = encoder()
        references[name] = (reference[:, :448] if args.publication else reference).cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stream = torch.cuda.Stream(device=device)
    print("Compile and capture complete vision catalog", flush=True)
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=torch.cuda.graph_pool_handle(), compile_decoder=True, compile_kinds=("vision",),
        cache_root=args.cache_root, before_run=lambda name: restore_inputs(name, values[name]))
    torch.cuda.synchronize()
    preparation_peak = torch.cuda.max_memory_allocated() - baseline
    report = dict(precision="replicated BF16", publication=args.publication, checkpoint=str(args.checkpoint),
                  preparation_peak_additional_bytes=preparation_peak, cases=[])
    try:
        for name in [*encoders, "vision65536", "vision256"]:
            encoder = encoders[name]
            restore_inputs(name, values[name])
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            compiled[name]()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            actual = observed(name).cpu()
            expected = references[name]
            equal = torch.equal(actual, expected)
            difference = actual.float() - expected.float()
            error = float(difference.norm() / expected.float().norm())
            row = dict(graph=name, bitwise_equal=equal, relative_l2=error, warm_peak_additional_bytes=peak)
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            assert equal, row
            restore_inputs(name, values[name])
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            for _ in range(3):
                graphs[name].replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            replayed = observed(name).cpu()
            replay_difference = replayed.float() - expected.float()
            replay_row = dict(graph=name, phase="captured replay",
                bitwise_equal=torch.equal(replayed, expected),
                mismatches=int((replayed != expected).sum()),
                maximum_absolute=float(replay_difference.abs().max()),
                relative_l2=float(replay_difference.norm()/expected.float().norm()))
            report["cases"].append(replay_row)
            print(json.dumps(replay_row), flush=True)
            if not replay_row["bitwise_equal"]:
                restore(encoder, values[name])
                repeated = encoder()
                repeated = (repeated[:, :448] if args.publication else repeated).cpu()
                print(json.dumps(dict(graph=name, phase="eager repeat",
                    relative_l2=float((repeated.float()-expected.float()).norm()/expected.float().norm()),
                    bitwise_equal=torch.equal(repeated, expected))), flush=True)
            assert replay_row["bitwise_equal"], replay_row
        # Non-dyadic interpolation factors and an inactive tail reuse the same graph.
        name = "vision65536"
        changed = fixture(250, 262, 1, device)
        encoder = encoders[name]
        restore_inputs(name, changed)
        expected = encoder()[:65500//4]
        expected = (expected[:, :448] if args.publication else expected).cpu()
        restore_inputs(name, changed)
        graphs[name].replay()
        torch.cuda.synchronize()
        actual = observed(name)[:65500//4].cpu()
        equal = torch.equal(actual, expected)
        row = dict(graph=name, active_patches=65500, bitwise_equal=equal,
                   relative_l2=float((actual.float()-expected.float()).norm()/expected.float().norm()))
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        assert equal, row
    finally:
        args.output.write_text(json.dumps(report, indent=2))
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection-cache", type=Path, help="reuse the namespace of a compatible deployment selection cache")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication", action="store_true")
    arguments = parser.parse_args()
    directory = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = directory + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, directory)
    from kimi31_vision_memory_probe import run
    run(arguments)
