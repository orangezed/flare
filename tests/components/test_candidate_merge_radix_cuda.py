"""FLARE_CANDIDATE_MERGE_RADIX replaces row top-k, threshold and cutoff counts with one radix program per row.

Output (selected IDs and their order) must equal the default six-launch merge for every input, including
integer ties, narrow score bands, all-equal rows, -inf / signed-zero / NaN scores and poisoned scratch.
"""
import os

import pytest
import torch

from ignition.components.candidate_merge import PreparedCandidateMerge
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(
    os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires one assigned CUDA GPU")


def _scores(kind, count, device, generator):
    if kind == "distinct":
        return torch.randn(count, device=device, generator=generator)
    if kind == "ties":
        return torch.randint(-3, 4, (count,), device=device, generator=generator).float()
    if kind == "band":
        return torch.rand(count, device=device, generator=generator) * 0.25 + 1.0
    if kind == "equal":
        return torch.full((count,), 0.5, device=device)
    scores = torch.randint(-1, 2, (count,), device=device, generator=generator).float()
    special = torch.rand(count, device=device, generator=generator)
    scores[special < 0.3] = -torch.inf
    scores[(special >= 0.3) & (special < 0.4)] = -0.0
    scores[(special >= 0.4) & (special < 0.42)] = float("nan")
    scores[(special >= 0.42) & (special < 0.44)] = -float("nan")
    return scores


@pytest.mark.parametrize("rows,width,topk", [(1, 16384, 2048), (5, 16384, 2048), (8, 8192, 2048),
                                             (3, 3000, 1024), (4, 16384, 512)])
def test_radix_cutoff_matches_default_merge(rows, width, topk, monkeypatch):
    device = torch.device("cuda:0")
    packed = torch.zeros((rows, width, 2), dtype=torch.float32, device=device)
    packed.view(torch.int32)[:, :, 1].fill_(-1)
    with RecordedPreparationSession(device=device, checkpoint="candidate-merge-radix-fixture",
            workload_schema="finite-score-candidates", numerical_policy="fp32", compile_workers=2) as session:
        workspace = ScratchArena(device)
        session.register_scratch(workspace)
        operations = {}
        for flag in ("0", "1"):
            monkeypatch.setenv("FLARE_CANDIDATE_MERGE_RADIX", flag)
            output = torch.empty((rows, topk), dtype=torch.int32, device=device)
            operations[flag] = (PreparedCandidateMerge(packed, output, session=session, workspace=workspace,
                                                       name="merge" + flag), output)
        session.tune(show_progress=False)
        assert operations["0"][0].plan.component_id == "flare.candidate_merge"
        assert operations["1"][0].plan.component_id == "flare.candidate_merge_radix"
        graphs = {}
        for flag, (operation, _) in operations.items():
            forward = compile_forward(operation)
            forward()
            graphs[flag] = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graphs[flag]):
                forward()
        session.freeze()
        for valid in sorted({0, 1, topk - 1, topk, topk + 1, 5000, width}):
            for kind in ("distinct", "ties", "band", "equal", "special"):
                generator = torch.Generator(device=device).manual_seed(97 * valid + len(kind))
                packed[:, :, 0].fill_(-torch.inf)
                packed.view(torch.int32)[:, :, 1].fill_(-1)
                for row in range(rows):
                    count = max(0, min(width, valid - row * 5))
                    slots = torch.randperm(width, device=device, generator=generator)[:count]
                    ids = torch.randperm(width, device=device, generator=generator)[:count]
                    packed[row, slots, 0] = _scores(kind, count, device, generator)
                    packed.view(torch.int32)[row, slots, 1] = ids.int()
                results = []
                for flag, (operation, output) in operations.items():
                    operation.scratch.fill_(255)
                    output.fill_(-777)
                    graphs[flag].replay()
                    torch.cuda.synchronize()
                    results.append(output.clone())
                assert torch.equal(results[0], results[1]), (valid, kind)
        for graph in graphs.values():
            graph.reset()


def test_radix_switch_keeps_default_merge_above_its_width(monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setenv("FLARE_CANDIDATE_MERGE_RADIX", "1")
    with RecordedPreparationSession(device=device, checkpoint="candidate-merge-radix-width",
            workload_schema="finite-score-candidates", numerical_policy="fp32", compile_workers=2) as session:
        workspace = ScratchArena(device)
        session.register_scratch(workspace)
        packed = torch.zeros((1, 16385, 2), dtype=torch.float32, device=device)
        output = torch.empty((1, 2048), dtype=torch.int32, device=device)
        operation = PreparedCandidateMerge(packed, output, session=session, workspace=workspace, name="wide")
        assert operation.plan.component_id == "flare.candidate_merge"
