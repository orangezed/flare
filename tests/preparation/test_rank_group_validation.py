# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rank readiness validates export templates without creating native executables."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


class Model:
    def __init__(self, rank, retained):
        self.retained = retained
        self.memory_report = {"rank": rank}

    def manifest(self):
        assert not self.retained, "export readiness must not instantiate the catalog"
        self.memory_report["validated"] = "executable"
        return {}

    def export_manifest(self):
        assert self.retained
        self.memory_report["validated"] = "template"
        return {}

    def close(self):
        pass


def prepare(snapshot, config, *, rank, group, progress=None):
    return Model(rank, config.retain_graphs)


@pytest.mark.parametrize("retained", [False, True])
def test_rank_readiness_chooses_the_requested_resource_kind(retained):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(int(retained))],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    from ignition.preparation.rank_group import prepare_group

    retained = bool(int(sys.argv[1]))
    config = SimpleNamespace(devices=(0, 1), retain_graphs=retained)
    with prepare_group("unused", config, prepare) as prepared:
        reports = prepared.memory_report["ranks"]
        assert [report["rank"] for report in reports] == [0, 1]
        assert all(report["validated"] == ("template" if retained else "executable") for report in reports)
