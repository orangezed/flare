# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.frontend import DiagnosticAssets


def test_assets_copy_exact_bytes_and_preserve_sources(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"prefix-template-suffix")
    text = "Tokenizer: \u017elu\u0165ou\u010dk\u00fd\n"
    owner = DiagnosticAssets({
        "tokenizer": {"text": text},
        "chat_template": {"path": str(source), "offset": 7, "nbytes": 8,
                          "sha256": hashlib.sha256(b"template").hexdigest().upper()},
    })
    source.write_bytes(b"changed source")
    paths = []
    for name, expected in (("tokenizer", text.encode()), ("chat_template", b"template")):
        descriptor = owner.descriptors[name]
        path = Path(descriptor["path"])
        paths.append(path)
        assert path.read_bytes() == expected
        assert descriptor["sha256"] == hashlib.sha256(expected).hexdigest()
    owner.close()
    assert all(not path.exists() for path in paths)
    assert source.read_bytes() == b"changed source"


@pytest.mark.parametrize("changes", [
    {"text": "also inline"}, {"offset": 1}, {"offset": -1, "nbytes": 2},
    {"offset": 0, "nbytes": 100}, {"sha256": "00" * 32},
])
def test_invalid_asset_descriptors_are_rejected(tmp_path, changes):
    source = tmp_path / "source"
    source.write_text("template")
    with pytest.raises(ValueError):
        DiagnosticAssets({"tokenizer": {"path": str(source), **changes}})


def test_debug_server_retains_files_until_native_drain(monkeypatch):
    import flare_heliopause
    from ignition.preparation.model import PreparedModel
    from tests.preparation.test_execution import _execution

    captured = {}
    runtime = SimpleNamespace(drained=False, close=lambda: None)
    def start(owner, manifest, config):
        captured.update(json.loads(config))
        return runtime

    monkeypatch.setattr(flare_heliopause, "Server", start, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    model = _execution(model_class=PreparedModel,
        server_config={"assets": {"tokenizer": {"text": "{}"}}}, memory_report={},
        session=SimpleNamespace(close=lambda: None),
        graphs={"decode": SimpleNamespace(reset=lambda: None),
                "prefill": SimpleNamespace(reset=lambda: None)})
    monkeypatch.setattr(model, "manifest", lambda **kwargs: {})
    model.start_server()
    path = Path(captured["assets"]["tokenizer"]["path"])
    assert path.read_text() == "{}"
    with pytest.raises(RuntimeError, match="did not drain"):
        model.close()
    assert path.exists()
    runtime.drained = True
    model.close()
    assert not path.exists()
