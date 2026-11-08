"""Preparation release and native serving drain retain separate bounded deadlines."""

from unittest.mock import Mock

import pytest
import torch.distributed as dist

from ignition.preparation.rank_group import PreparedRankGroup


@pytest.mark.parametrize("started,deadline", [(False, 300000), (True, 60000)])
def test_cleanup_deadline_preserves_serving_drain(monkeypatch, started, deadline):
    destroyed = Mock()
    monkeypatch.setattr(dist, "destroy_process_group", destroyed)
    group = PreparedRankGroup.__new__(PreparedRankGroup)
    group._closed, group._started = False, started
    group.supervisor, group.model, group.rendezvous = Mock(), Mock(), Mock()
    group.controls, group.connections, group.processes = [Mock()], [Mock()], [Mock()]
    group.close()
    group.supervisor.begin_shutdown.assert_called_once_with(deadline)
    group.model.close.assert_called_once_with()
    destroyed.assert_called_once_with()
    group.supervisor.finish_shutdown.assert_called_once_with()
    group.processes[0].join.assert_called_once_with()
    group.supervisor.abort.assert_not_called()
    if started:
        group.controls[0].send.assert_not_called()
    else:
        group.controls[0].send.assert_called_once_with("close")
    assert group._closed
    group.close()
    group.supervisor.begin_shutdown.assert_called_once_with(deadline)
