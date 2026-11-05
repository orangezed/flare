# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain disk inputs to native package combination without changing model work.

Run with the normal preparation interpreter and dependency paths:
  python qualify/runtime_package/capture_sources.py --capture-dir /data/export-inputs -- export ...
The retained inputs omit publisher requests and signing material. Files are
hardlinked on the same filesystem; no tensor or weight bytes are copied.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def capture_sources(ranks, peers, cache, destination, *, form, cache_only):
    destination = Path(destination).resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    files = destination / 'files'
    blobs = destination / 'blobs'
    files.mkdir(); blobs.mkdir()
    linked = {}
    saved = []
    ids = set()

    def retain_file(source):
        source = Path(source).resolve(strict=True)
        stamp = source.stat()
        if not source.is_file():
            raise ValueError(f'export source is not a regular file: {source}')
        identity = (stamp.st_dev, stamp.st_ino)
        if identity not in linked:
            target = files / f'{len(linked):08d}.data'
            os.link(source, target, follow_symlinks=True)
            linked[identity] = target
        return str(linked[identity])

    def payload_id(value):
        if not value:
            return
        if not isinstance(value, list) or len(value) != 32 or any(type(x) is not int or not 0 <= x < 256 for x in value):
            raise ValueError('invalid cached payload identity')
        ids.add(bytes(value).hex())

    def engram_ids(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'payload_id': payload_id(item)
                else: engram_ids(item)
        elif isinstance(value, list):
            for item in value: engram_ids(item)

    for ordinal, source in enumerate(ranks):
        record = json.loads(Path(source).read_text())
        for part in record['payload']:
            part['source']['path'] = retain_file(part['source']['path'])
        weights = record.get('weight_payload') or {}
        for group in [*weights.get('groups', []), weights.get('initial_state', {})]:
            payload_id(group.get('payload_id'))
        engram_ids(weights.get('engram'))
        target = destination / f'rank-{ordinal}.json'
        target.write_text(json.dumps(record, separators=(',', ':')))
        saved.append(str(target))
    for identity in sorted(ids):
        source = Path(cache) / 'blobs' / f'{identity}.safetensors'
        os.link(source, blobs / source.name)
    request = dict(ranks=saved, peer_memory=peers, cache=str(cache), form=form,
                   cache_only=cache_only, cached_payload_ids=sorted(ids))
    (destination / 'combine-inputs.json').write_text(json.dumps(request, indent=2))
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('ignition_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.ignition_args
    if arguments[:1] == ['--']: arguments = arguments[1:]
    if arguments[:1] != ['export']:
        parser.error('the wrapped command must be ignition export')
    if args.capture_dir.exists(): parser.error('capture directory must not exist')
    from ignition.cli import environment, main as ignition_main
    original_apply = environment.apply
    installed = False

    def apply_and_observe(*values, **keywords):
        nonlocal installed
        original_apply(*values, **keywords)
        if installed: return
        installed = True
        from ignition.preparation import model_package, package_file
        original_combine = model_package.combine

        def combine(path, ranks, peer_memory, **options):
            cache = package_file._options(path)['cache']
            try:
                capture_sources(ranks, peer_memory, cache, args.capture_dir,
                                form=options.get('form', 'directory'),
                                cache_only=options.get('cache_only', False))
                print(f'Retained package-combination inputs: {args.capture_dir}', flush=True)
            except Exception as error:
                print(f'Package source capture failed; normal export continues: {error}', file=sys.stderr, flush=True)
            return original_combine(path, ranks, peer_memory, **options)

        model_package.combine = combine

    environment.apply = apply_and_observe
    return ignition_main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
