#!/usr/bin/env python3
"""Explicit allowlist export of completed experimental phases; never whole-workspace copy."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clef_snake.audit import audit


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k not in {"model_path", "head_path", "phase_artifacts", "source_result_sha256"}}
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, str) and ("/Users/" in value or "/Volumes/" in value):
        return "[local path omitted]"
    return value


def compressed(path, raw):
    assert b"/Users/" not in raw and b"/Volumes/" not in raw, "Unexpected private path in public inference evidence"
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
            stream.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--iq2", type=Path, required=True)
    parser.add_argument("--q2", type=Path)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    phases = [("primary", args.primary), ("iq2", args.iq2)]
    if args.q2:
        phases.append(("q2", args.q2))
    output = args.out
    for name in ["decisions", "server-logs", "provenance"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    result = json.loads((args.combined / "results.json").read_text())
    dump(output / "results.json", public(result))
    dump(output / "rounds.json", result["rounds"])
    (output / "seed-manifest.json").write_bytes((args.primary / "seed-manifest.json").read_bytes())
    dump(output / "schedule.json", [{k: r[k] for k in ["sequence", "model", "seed_round", "seed"]} for r in result["rounds"]])
    with (output / "rounds.csv").open("w", newline="") as stream:
        fields = ["sequence", "model", "seed_round", "seed", "score", "successful_moves", "attempted_moves", "end_reason", "alive"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in result["rounds"])
    wanted = {}
    inventories = {}
    for phase, path in phases:
        assert json.loads((path / "audit.json").read_text())["passed"]
        raw = (path / "decisions.jsonl").read_bytes()
        compressed(output / "decisions" / f"{phase}.jsonl.gz", raw)
        for line in raw.splitlines():
            call = json.loads(line)
            wanted.setdefault(call["model"], set()).add(call["response"]["provenance"]["inference_id"])
        config = json.loads((path / "config.json").read_text())
        for name, digest in config["source_sha256"].items():
            assert sha(path / "sources" / name) == digest, name
            assert sha(args.runtime / name) == digest, name
        inventories[phase] = {"original_decisions_jsonl_sha256": sha(path / "decisions.jsonl"),
                              "original_results_json_sha256": sha(path / "results.json"),
                              "executed_source_sha256": config["source_sha256"],
                              "source_archive_locally_verified": True,
                              "source_note": "Original UI is not redistributed: upstream has no explicit license. Portable game parity is verified against every recorded request and transition."}
    locations = {"bf16": "decisions.jsonl", "q6-k-l": "quant/decisions.jsonl", "q4-k-m": "q4/decisions.jsonl",
                 "iq2-m": "iq2/decisions.jsonl", "q2-k": "q2/decisions.jsonl"}
    for model, ids in wanted.items():
        kept = []
        with (args.runtime / locations[model]).open("rb") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("response", {}).get("provenance", {}).get("inference_id") in ids:
                    kept.append(line)
        assert len(kept) == len(ids)
        compressed(output / "server-logs" / f"{model}.jsonl.gz", b"".join(kept))
    for variant in ["quant", "q4", "iq2"] + (["q2"] if args.q2 else []):
        for name in ["provenance.json", "embedding-alignment.json", "header-inspection.json"]:
            source = args.runtime / variant / name
            if source.exists():
                dump(output / "provenance" / f"{variant}-{name}", public(json.loads(source.read_text())))
    dump(output / "provenance" / "bridge-validation.json", public(json.loads((args.runtime / "quant/bridge-validation.json").read_text())))
    dump(output / "evidence-inventory.json", inventories)
    combined_audit = json.loads((args.combined / "audit.json").read_text())
    dump(output / "original-audit.json", public(combined_audit))
    proof = audit(output)
    assert proof["server_request_response_exact"]
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(output).as_posix()}\n" for p in files))
    print(json.dumps({"output": str(output), "rounds": proof["rounds"], "calls": proof["calls"], "passed": proof["passed"]}), flush=True)


if __name__ == "__main__":
    main()
