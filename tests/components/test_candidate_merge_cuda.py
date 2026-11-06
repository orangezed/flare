"""Candidate merging uses prepared programs and preserves exact top-k scores."""
import os

import pytest
import torch

from ignition.components.candidate_merge import PreparedCandidateMerge
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(
    os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires one assigned CUDA GPU")


@pytest.mark.parametrize("rows,width", [(1, 16384), (4, 16384), (32, 16384),
                                      (1, 32768), (4, 32768), (32, 33792)])
def test_prepared_merge_replays_mutable_scattered_candidates(rows, width):
    device, topk = torch.device("cuda:0"), 2048
    packed = torch.empty((rows, width, 2), dtype=torch.float32, device=device)
    output = torch.empty((rows, topk), dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="candidate-merge-fixture",
            workload_schema="finite-score-candidates", numerical_policy="fp32", compile_workers=2) as session:
        workspace = ScratchArena(device)
        session.register_scratch(workspace)
        operation = PreparedCandidateMerge(packed, output, session=session, workspace=workspace, name="merge")
        session.tune(show_progress=False)
        packed[:, :, 0].zero_()
        packed.view(torch.int32)[:, :, 1].fill_(-1)
        forward = compile_forward(operation)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for valid in (0, 1, 221, 2047, 2048, 2049, 4096, width):
            for ties in (False, True):
                torch.manual_seed(771+valid)
                packed[:, :, 0].fill_(-torch.inf)
                packed.view(torch.int32)[:, :, 1].fill_(-1)
                expected = []
                for row in range(rows):
                    count = max(0, valid-row*3)
                    ids = torch.randperm(width, device=device)[:count]
                    slots = torch.randperm(width, device=device)[:count]
                    scores = torch.randint(-3, 4, (count,), device=device).float() if ties else ids.float()/65536
                    packed[row, slots, 0] = scores
                    packed.view(torch.int32)[row, slots, 1] = ids.int()
                    chosen = scores.topk(min(count, topk)).indices
                    expected.append((ids, scores, chosen))
                operation.scratch.fill_(255)
                output.fill_(-777)
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                for row, (ids, scores, chosen) in enumerate(expected):
                    result = output[row][output[row] >= 0].long()
                    assert result.numel() == min(ids.numel(), topk)
                    assert result.unique().numel() == result.numel()
                    if not ties:
                        torch.testing.assert_close(result.sort().values, ids[chosen].sort().values, atol=0, rtol=0)
                    else:
                        lookup = torch.full((width,), torch.nan, device=device)
                        lookup[ids] = scores
                        torch.testing.assert_close(lookup[result].sort().values,
                                                   scores[chosen].sort().values, atol=0, rtol=0)
                        slots = (packed.view(torch.int32)[row, :, 1] >= 0).nonzero().flatten()
                        order = packed[row, slots, 0].argsort(descending=True, stable=True)[:topk]
                        expected_ids = packed.view(torch.int32)[row, slots[order], 1].long()
                        torch.testing.assert_close(result.sort().values, expected_ids.sort().values,
                                                   atol=0, rtol=0)
        graph.reset()
