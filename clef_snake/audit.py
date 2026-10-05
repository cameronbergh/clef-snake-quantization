"""Offline audit: exact browser request parity, trajectories, food and server logs."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from .benchmark import choice
from .game import Game


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def ordered(value):
    # Also verify key order: schema serialization can influence tokenization.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def audit(root):
    manifest = json.loads((root / "seed-manifest.json").read_text())
    seeds = manifest["seeds"]
    assert seeds and len(seeds) == len(set(seeds)), "Saved seeds must be distinct"
    published = json.loads((root / "results.json").read_text())
    assert published["seed_manifest"] == manifest
    expected_models = set(published["aggregate_by_model"])
    round_summaries = json.loads((root / "rounds.json").read_text())
    assert len(round_summaries) == published["total_rounds"] == len(expected_models) * len(seeds)
    assert set(r["model"] for r in round_summaries) == expected_models
    grouped = defaultdict(list)
    paths = sorted((root / "decisions").glob("*.jsonl.gz"))
    if not paths:
        paths = sorted((root / "decisions").glob("*.jsonl"))
    for path in paths:
        for record in rows(path):
            grouped[(record["model"], record["seed_round"])].append(record)
    assert len(grouped) == len(round_summaries)
    server = {}
    for path in sorted((root / "server-logs").glob("*.jsonl.gz")):
        model = path.name.removesuffix(".jsonl.gz")
        server[model] = {}
        for record in rows(path):
            iid = record["response"]["provenance"]["inference_id"]
            assert iid not in server[model]
            server[model][iid] = record
    assert set(server) == expected_models, "Independent server logs are required for every published model"
    assert {m for m, number in grouped} == expected_models
    ids = defaultdict(list)
    first_requests = defaultdict(dict)
    checks = []
    for summary in round_summaries:
        model, number = summary["model"], summary["seed_round"]
        seed = seeds[number - 1]
        assert seed == summary["seed"]
        records = grouped[(model, number)]
        assert len(records) == summary["attempted_moves"]
        game = Game(seed)
        for attempt, record in enumerate(records, 1):
            assert record["attempt"] == attempt and record["seed"] == seed
            assert ordered(record["request"]) == ordered(game.request()), (model, number, attempt, "request/schema/feature/order drift")
            if attempt == 1:
                first_requests[number][model] = record["request"]
            response = record["response"]
            iid = response["provenance"]["inference_id"]
            ids[model].append(iid)
            move = choice(response, model)
            game.apply(move)
            assert all(record["after"][k] == v for k, v in game.snapshot().items()), (model, number, attempt, "transition drift")
            assert len(record["food_events"]) == len(game.food_events)
            for recorded, expected in zip(record["food_events"], game.food_events):
                assert all(recorded[k] == v for k, v in expected.items()), (model, number, attempt, "food drift")
            if model in server:
                actual = server[model][iid]
                assert actual["request"] == record["request"] and actual["response"] == response
            if not game.alive:
                assert attempt == len(records)
        assert summary["score"] == game.score and summary["successful_moves"] == game.steps
        assert summary["alive"] == game.alive
        assert summary["end_reason"] == ("move_cap" if game.alive else "collision")
        if game.alive:
            assert game.steps == 500
        checks.append({"model": model, "seed_round": number, "calls": len(records), "score": game.score})
    for model, values in ids.items():
        assert len(values) == len(set(values))
        assert all(b == a + 1 for a, b in zip(values, values[1:])), model
        if model in server:
            assert set(values) == set(server[model])
    for group in first_requests.values():
        requests = list(group.values())
        assert all(ordered(r) == ordered(requests[0]) for r in requests)
    counts = {m: sum(r["model"] == m for r in checks) for m in ids}
    assert set(counts) == expected_models
    assert all(value == len(seeds) for value in counts.values())
    assert all(counts[m] == published["aggregate_by_model"][m]["rounds"] for m in counts)
    for model in counts:
        scores = [r["score"] for r in checks if r["model"] == model]
        aggregate = published["aggregate_by_model"][model]
        assert aggregate["scores"]["n"] == len(scores)
        assert aggregate["scores"]["mean"] == round(statistics.mean(scores), 3)
        assert aggregate["scores"]["median"] == statistics.median(scores)
        assert (aggregate["scores"]["min"], aggregate["scores"]["max"]) == (min(scores), max(scores))
        assert aggregate["score_std_population"] == round(statistics.pstdev(scores), 3)
        assert aggregate["total_model_calls"] == len(ids[model])
    result = {"passed": True, "rounds": len(checks), "paired_seed_units": len(seeds), "calls": sum(len(v) for v in ids.values()),
              "rounds_by_model": counts, "exact_request_and_field_order_parity": True, "all_trajectory_and_food_events_verified": True,
              "all_head_argmax_and_provenance_verified": True, "no_duplicate_or_competing_inference_ids": True,
              "server_request_response_exact": set(server) == set(ids), "episodes": checks}
    (root / "portable-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_hashes(root):
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = root / name
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        with path.open("rb") as stream:
            actual = hashlib.sha256()
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                actual.update(chunk)
            assert actual.hexdigest() == digest, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.verify_hashes:
        verify_hashes(args.data)
    result = audit(args.data)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
