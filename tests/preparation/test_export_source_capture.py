# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Retained publisher inputs preserve exact disk sources without model work."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'qualify/runtime_package/capture_sources.py'
spec = importlib.util.spec_from_file_location('capture_package_sources', SCRIPT)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def sources(tmp_path, *, symlink=False):
    stage = tmp_path / 'stage'
    stage.mkdir()
    cache = tmp_path / 'cache'
    (cache / 'blobs').mkdir(parents=True)
    data = stage / 'module'
    data.write_bytes(b'compiled-resource')
    source = data
    if symlink:
        source = stage / 'module-link'
        source.symlink_to(data)
    identity = hashlib.sha256(b'weight-data').digest()
    blob = cache / 'blobs' / f'{identity.hex()}.safetensors'
    blob.write_bytes(b'weight-data')
    record = dict(index={'version': 1, 'opaque': 'preserved'},
                  payload=[dict(offset=12, source=dict(path=str(source), offset=2, nbytes=4))],
                  weight_payload=dict(groups=[dict(payload_id=list(identity))],
                                      initial_state=dict(payload_id=[])), hardware='sm120')
    ranks = []
    for rank in range(2):
        path = stage / f'rank-{rank}.json'
        path.write_text(json.dumps(record))
        ranks.append(path)
    return stage, cache, data, blob, record, ranks


@pytest.mark.parametrize('symlink', [False, True])
def test_capture_retains_resources_and_payloads_after_source_cleanup(tmp_path, symlink):
    stage, cache, data, blob, original, ranks = sources(tmp_path, symlink=symlink)
    out = tmp_path / 'retained'
    request = capture.capture_sources(ranks, [dict(owner=0, address=123)], cache, out,
                                      form='directory', cache_only=True)
    assert len(list((out / 'files').iterdir())) == 1
    retained = next((out / 'files').iterdir())
    assert retained.samefile(data) and not retained.is_symlink()
    assert (out / 'blobs' / blob.name).samefile(blob)
    assert json.loads(ranks[0].read_text()) == original
    shutil.rmtree(stage)
    shutil.rmtree(cache)
    for rank in request['ranks']:
        value = json.loads(Path(rank).read_text())
        assert value['index'] == original['index']
        assert value['weight_payload'] == original['weight_payload']
        assert value['payload'][0]['offset'] == 12
        assert value['payload'][0]['source']['offset'] == 2
        assert value['payload'][0]['source']['nbytes'] == 4
        assert Path(value['payload'][0]['source']['path']).read_bytes() == b'compiled-resource'
    assert (out / 'blobs' / blob.name).read_bytes() == b'weight-data'
    assert set(request) == {'ranks', 'peer_memory', 'cache', 'form', 'cache_only', 'cached_payload_ids'}
    assert out.stat().st_mode & 0o777 == 0o700


def test_capture_resolves_output_paths_and_refuses_overwrite(tmp_path, monkeypatch):
    stage, cache, _, _, _, ranks = sources(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / 'retained'
    request = capture.capture_sources(ranks, [], cache, Path('retained'),
                                      form='directory', cache_only=False)
    assert all(Path(path).is_absolute() for path in request['ranks'])
    with pytest.raises(FileExistsError):
        capture.capture_sources(ranks, [], cache, out, form='directory', cache_only=False)
    assert all(path.exists() for path in ranks)


def test_capture_rejects_invalid_payload_id_without_changing_sources(tmp_path):
    _, cache, _, _, record, ranks = sources(tmp_path)
    record['weight_payload']['groups'][0]['payload_id'] = [256] * 32
    ranks[0].write_text(json.dumps(record))
    before = ranks[0].read_bytes()
    with pytest.raises(ValueError, match='invalid cached payload identity'):
        capture.capture_sources(ranks, [], cache, tmp_path / 'retained',
                                form='directory', cache_only=True)
    assert ranks[0].read_bytes() == before
