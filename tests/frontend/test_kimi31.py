# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native K3.1 XTML rendering matches its checkpoint encoder on normalized messages."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_prompt_bytes_and_tokens(tmp_path):
    checkpoint = os.getenv("FLARE_KIMI31_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set FLARE_KIMI31_CHECKPOINT to checkpoint tokenizer and encoder assets")
    checkpoint = Path(checkpoint)
    binary = Path(os.getenv("FLARE_TEST_FRONTEND", str(ROOT / "helios/target/debug/test_frontend")))
    if not binary.is_file():
        pytest.skip("build test_frontend with frontend,test-tools features")
    import tiktoken
    from ignition.recipes.kimi3.tokenizer import PATTERN, _Converter, convert

    spec = importlib.util.spec_from_file_location("kimi31_encoder", checkpoint / "encoding_k3.py")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    spec.loader.exec_module(encoder)
    tokenizer = convert(checkpoint)
    vocabulary = tiktoken.Encoding(name="kimi31", pat_str=PATTERN,
        mergeable_ranks=_Converter.load_tiktoken_bpe(checkpoint / "tiktoken.model"),
        special_tokens={tokenizer.id_to_token(i): i for i in range(163584, 163840)})
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    shutil.copyfile(checkpoint / "tokenizer_config.json", tmp_path / "tokenizer_config.json")
    shutil.copyfile(ROOT / "ignition/recipes/kimi31/assets/chat_template.jinja", tmp_path / "chat_template.jinja")
    fixtures = dict(template_encoding=1, cases=[], encode_cases=[], decode_cases=[], assets={
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()})
    fixtures["template_defaults"] = {"add_generation_prompt": True}
    fixtures["default_overrides"] = [("text-max", {}), ("text-low", {"reasoning_effort": "low"}),
                                      ("text-medium", {"reasoning_effort": "medium"})]
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Read a document",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
    call = {"id": "call-a", "function": {"name": "lookup", "arguments": {
        "query": 'café 中文 <|close|> "quoted"', "count": 3, "ratio": 1.5, "flag": True,
        "empty": None, "list": [1, "x"], "object": {"z": False, "a": []}}}}
    conversations = [
        ("text", [{"role": "user", "content": "Hello"}], {}),
        ("literal", [{"role": "user", "content": '<|open|>think<|sep|>\n你好'}], {}),
        ("history", [{"role": "system", "content": "Be precise"}, {"role": "user", "content": "Hello"},
            {"role": "assistant", "reasoning_content": "Let me think", "content": "Hi"},
            {"role": "user", "content": "Continue"}], {}),
        ("whitespace-reasoning", [{"role": "user", "content": "Hello"},
            {"role": "assistant", "reasoning_content": " \n ", "content": "Hi"},
            {"role": "user", "content": "Continue"}], {}),
        # The shared HTTP decoder parses string tool arguments into objects before rendering.
        ("tools", [{"role": "user", "content": "Search"}, {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call-a", "content": "Found"}], {"tools": tools}),
        ("required", [{"role": "user", "content": "Search"}], {"tools": tools, "tool_choice": "required"}),
        ("image", [{"role": "user", "content": [{"type": "text", "text": "Inspect "},
            {"type": "image_url", "image_url": {"url": "unused"}}]}], {}),
    ]
    for effort in ("low", "medium", "high", "max"):
        for name, messages, extra in conversations:
            options = dict(add_generation_prompt=True, reasoning_effort=effort, **extra)
            args = dict(add_generation_prompt=True, thinking=True, thinking_effort=effort, **extra)
            if name == "image":
                args["image_prompts"] = ["<|media_pad|>"]
            segments = encoder.build_chat_segments(messages, **args)
            token_ids = []
            for segment in segments:
                token_ids.extend(vocabulary.encode(segment.text,
                    allowed_special="all" if segment.allow_special else set(), disallowed_special=()))
            fixtures["cases"].append(dict(name=f"{name}-{effort}", messages=messages, options=options,
                rendered="".join(s.text for s in segments), token_ids=token_ids))
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(fixtures, ensure_ascii=False))
    result = subprocess.run([str(binary), str(tmp_path), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
