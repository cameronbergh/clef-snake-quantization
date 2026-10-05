"""Serial model-backed evaluation. Requires an explicit saved seed manifest."""
import argparse
import csv
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from .catalog import MODELS, OFFICIAL_REPO, OFFICIAL_REV, QUANT_REV
from .game import Game, DIRECTIONS


def http(base, endpoint, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(base + endpoint, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def verify_provenance(value, model):
    assert value["repo"] == OFFICIAL_REPO and value["revision"] == OFFICIAL_REV
    assert value["fallback"] is False and value["safety_override"] is False
    if model != "bf16":
        assert value["quantization"] == MODELS[model]["quantization"]
        assert value["quant_revision"] == QUANT_REV and value["quant_sha256"] == MODELS[model]["sha256"]


def choice(response, model):
    verify_provenance(response["provenance"], model)
    answer = response["answers"]["move"]
    probabilities = answer["probabilities"]
    assert set(probabilities) == set(DIRECTIONS)
    selected = answer["choice"]
    assert selected in DIRECTIONS and probabilities[selected] == max(probabilities.values())
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--schedule", type=Path, help="Explicit published execution order; otherwise rotate requested models per seed")
    parser.add_argument("--endpoints", type=Path, help="JSON model-name to base-URL mapping")
    parser.add_argument("--move-cap", type=int, default=500)
    args = parser.parse_args()
    if args.move_cap < 1:
        parser.error("--move-cap must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output must be empty: completed evidence is never overwritten")
    manifest = json.loads(args.manifest.read_text())
    seeds = manifest["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Manifest must contain distinct saved seeds")
    endpoints = {m: f"http://127.0.0.1:{v['port']}" for m, v in MODELS.items()}
    if args.endpoints:
        endpoints.update(json.loads(args.endpoints.read_text()))
    if args.schedule:
        schedule = json.loads(args.schedule.read_text())
    else:
        schedule = [{"model": m, "seed_round": i + 1, "seed": seed} for i, seed in enumerate(seeds)
                    for m in args.models[i % len(args.models):] + args.models[:i % len(args.models)]]
    expected = {(m, i + 1) for m in args.models for i in range(len(seeds))}
    assert {(r["model"], r["seed_round"]) for r in schedule} == expected
    assert len(schedule) == len(expected)
    assert all(r["seed"] == seeds[r["seed_round"] - 1] for r in schedule)
    for model in args.models:
        status = http(endpoints[model], "/health")
        assert status["state"] == "ready", status
        verify_provenance(status, model)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "seed-manifest.json").write_bytes(args.manifest.read_bytes())
    (args.out / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    (args.out / "config.json").write_text(json.dumps({"move_cap": args.move_cap, "driver": "portable-python-v1", "models": args.models}, indent=2) + "\n")
    rounds = []
    with (args.out / "decisions.jsonl").open("w") as log:
        for sequence, scheduled in enumerate(schedule, 1):
            model = scheduled["model"]
            game = Game(scheduled["seed"])
            before = http(endpoints[model], "/health")["inferences"]
            attempt = 0
            start = time.perf_counter()
            while game.alive and game.steps < args.move_cap:
                attempt += 1
                body = game.request()
                t0 = time.perf_counter()
                response = http(endpoints[model], "/v1/systemone", body)
                wall_ms = (time.perf_counter() - t0) * 1000
                game.apply(choice(response, model))
                row = {"sequence": sequence, **scheduled, "attempt": attempt, "request": body,
                       "response": response, "request_wall_ms": wall_ms, "after": game.snapshot(),
                       "food_events": game.food_events}
                log.write(json.dumps(row) + "\n")
                log.flush()
            after = http(endpoints[model], "/health")["inferences"]
            assert after - before == attempt, "Another client used this model during the round"
            summary = {"sequence": sequence, **scheduled, "score": game.score, "successful_moves": game.steps,
                       "attempted_moves": attempt, "alive": game.alive, "end_reason": "move_cap" if game.alive else "collision",
                       "duration_s": round(time.perf_counter() - start, 3)}
            rounds.append(summary)
            (args.out / "completed-rounds.json").write_text(json.dumps(rounds, indent=2) + "\n")
            print(json.dumps({"completed": len(rounds), "total": len(schedule), **summary}), flush=True)
    with (args.out / "rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rounds[0]))
        writer.writeheader()
        writer.writerows(rounds)


if __name__ == "__main__":
    main()
