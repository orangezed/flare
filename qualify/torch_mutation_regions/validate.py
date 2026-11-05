"""Reproduce CPU parity and bounded timings in separate original/patched processes."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline-python", type=Path, required=True)
parser.add_argument("--patched-python", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
source = Path(__file__).resolve().parent
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
os.nice(15)
interpreters = {
    "baseline": str(args.baseline_python.absolute()),
    "patched": str(args.patched_python.absolute()),
}
for label, interpreter in interpreters.items():
    for script, suffix in (("cases.py", ""), ("extra_checks.py", "-extra")):
        stem = label + suffix
        command = [interpreter, "-B", str(source / script), "--output", str(args.output / (stem + ".json"))]
        with (args.output / (stem + ".log")).open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
baseline = json.loads((args.output / "baseline.json").read_text())
patched = json.loads((args.output / "patched.json").read_text())
assert baseline["default_barrier"] == patched["default_barrier"] == [1, 0]
assert not baseline["cuda_initialized"] and not patched["cuda_initialized"]
assert len(baseline["rows"]) == len(patched["rows"])
for old, new in zip(baseline["rows"], patched["rows"]):
    assert old["name"] == new["name"]
    for key in ("code", "regions", "custom", "output", "nodes"):
        assert old[key] == new[key], (old["name"], key)
for count in (32, 128, 384):
    old = [x for x in baseline["rows"] if x["name"].startswith(f"scale_{count}_")]
    new = [x for x in patched["rows"] if x["name"].startswith(f"scale_{count}_")]
    print(json.dumps(dict(rewrites=count, baseline_s=statistics.median(x["wall_s"] for x in old),
        patched_s=statistics.median(x["wall_s"] for x in new),
        baseline_scans=old[0]["scans"], patched_scans=new[0]["scans"])))
old = json.loads((args.output / "baseline-extra.json").read_text())
new = json.loads((args.output / "patched-extra.json").read_text())
for name in old:
    assert old[name]["graphs"] == new[name]["graphs"], name
    assert old[name]["output"] == new[name]["output"], name
    print(json.dumps(dict(case=name, baseline_s=old[name]["wall_s"], patched_s=new[name]["wall_s"],
        baseline_scans=old[name]["scans"], patched_scans=new[name]["scans"])))
print("PASS: all20cases have exact FX/region/context/output/alias parity; default mutation barriers preserved; CUDA uninitialized.")
