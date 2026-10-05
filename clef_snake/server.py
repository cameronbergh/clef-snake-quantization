"""Configurable loopback server; original official CLEF head, no safety override."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback
from .catalog import MODELS, OFFICIAL_REPO, OFFICIAL_REV, QUANT_REPO, QUANT_REV, NATIVE_REV, HEAD_SHA, HEAD_SOURCE_SHA


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    official = args.official.expanduser().resolve()
    if args.model != "bf16" and (args.gguf is None or args.bridge is None):
        parser.error("Quantized models require --gguf and --bridge")
    assert sha(official / "joint_head.safetensors") == HEAD_SHA
    assert sha(official / "joint_schema_model.py") == HEAD_SOURCE_SHA
    if args.model != "bf16":
        assert args.gguf.stat().st_size == MODELS[args.model]["size"]
        assert sha(args.gguf) == MODELS[args.model]["sha256"]
    sys.path.insert(0, str(official))
    os.environ["CLEF_OFFICIAL_DIR"] = str(official)
    if args.bridge:
        os.environ["CLEF_BRIDGE_LIBRARY"] = str(args.bridge.resolve())
    import torch
    from fastapi import FastAPI, HTTPException
    import uvicorn
    assert torch.backends.mps.is_available(), "Published inference path requires macOS MPS"
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "CPU fallback must be disabled"
    status = {"state": "loading", "repo": OFFICIAL_REPO, "revision": OFFICIAL_REV,
              "device": "mps" if args.model == "bf16" else "Metal + MPS",
              "dtype": "bfloat16" if args.model == "bf16" else MODELS[args.model]["quantization"] + " backbone + bfloat16 head",
              "fallback": False, "safety_override": False, "inferences": 0}
    if args.model != "bf16":
        status.update(quantization=MODELS[args.model]["quantization"], quant_repo=QUANT_REPO, quant_revision=QUANT_REV,
                      quant_sha256=MODELS[args.model]["sha256"], native_revision=NATIVE_REV,
                      head="unchanged official JointSchemaHead", hidden_states="all tokens, causal final_norm, raw and unpooled",
                      output_embeddings="exact GGUF output.weight rows dequantized and cast BF16 for official head")
    engine = {}
    lock = threading.Lock()
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def load():
        try:
            if args.model == "bf16":
                from joint_schema_model import load_release_model, systemone
                model, processor = load_release_model(official, device="mps", dtype=torch.bfloat16,
                                                     local_files_only=True, attn_implementation="sdpa")
            else:
                from .model_bridge import load as load_quant, systemone
                model, processor = load_quant(args.gguf)
            torch.mps.synchronize()
            engine.update(model=model, processor=processor, systemone=systemone)
            status.update(state="ready", head_parameters=sum(p.numel() for p in model.head.parameters()))
            assert status["head_parameters"] == 121762820
        except Exception as error:
            status.update(state="error", error=str(error))
            traceback.print_exc()

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=load, daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return status.copy()

    def infer(body):
        if status["state"] != "ready":
            raise HTTPException(503, detail=status.copy())
        if body.get("model") != "clef-flash" or body.get("images") or body.get("videos"):
            raise HTTPException(400, detail="Text/JSON clef-flash requests only")
        with lock:
            start = time.perf_counter()
            try:
                response = engine["systemone"](engine["model"], engine["processor"], body, max_length=4096)
                torch.mps.synchronize()
            except Exception as error:
                traceback.print_exc()
                raise HTTPException(500, detail=str(error)) from error
            for answer in response["answers"].values():
                if not all(math.isfinite(v) for v in answer.get("probabilities", {}).values()):
                    raise HTTPException(500, detail="Non-finite official-head probabilities")
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            status["inferences"] += 1
            response["provenance"] = {k: v for k, v in status.items() if k not in {"state", "inferences", "head_parameters"}}
            response["provenance"]["inference_id"] = status["inferences"]
            timings = {"total_ms": elapsed}
            if args.model != "bf16":
                timings.update(engine["model"].timings)
            response["diagnostics"] = {"timing": timings}
            # New runs record this for all models. Published BF16 run did not.
            response["diagnostics"]["memory"] = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            with args.log.open("a") as stream:
                stream.write(json.dumps({"time": time.time(), "request": body, "response": response}) + "\n")
            return response

    @app.post("/v1/systemone")
    async def decision(body: dict):
        return await asyncio.to_thread(infer, body)

    uvicorn.run(app, host="127.0.0.1", port=args.port or MODELS[args.model]["port"])


if __name__ == "__main__":
    main()
