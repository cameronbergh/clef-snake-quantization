"""Download pinned official/quant artifacts and verify every payload locally."""
import argparse
import hashlib
import json
from pathlib import Path
from .catalog import MODELS, OFFICIAL_REPO, OFFICIAL_REV, QUANT_REPO, QUANT_REV, HEAD_SHA, HEAD_SOURCE_SHA


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--head-only", action="store_true", help="Skip official backbone shards when only running quants")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download, get_hf_file_metadata, hf_hub_url
    root = args.root.expanduser().resolve()
    official = root / "official"
    info = HfApi().model_info(OFFICIAL_REPO, revision=OFFICIAL_REV, files_metadata=True)
    files = [f for f in info.siblings if not f.rfilename.startswith(".") and
             not (args.head_only and f.rfilename.startswith("model-") and f.rfilename.endswith(".safetensors"))]
    assert not (args.head_only and "bf16" in args.models), "BF16 needs the official backbone"
    snapshot_download(OFFICIAL_REPO, revision=OFFICIAL_REV, local_dir=official, allow_patterns=[f.rfilename for f in files])
    verified = []
    for file in files:
        path = official / file.rfilename
        assert path.stat().st_size == file.size, file.rfilename
        digest = sha(path)
        if file.lfs:
            assert digest == file.lfs.sha256, file.rfilename
        verified.append({"file": file.rfilename, "size": path.stat().st_size, "sha256": digest})
    assert sha(official / "joint_head.safetensors") == HEAD_SHA
    assert sha(official / "joint_schema_model.py") == HEAD_SOURCE_SHA
    (official / "VERIFIED.json").write_text(json.dumps({"repo": OFFICIAL_REPO, "revision": OFFICIAL_REV, "files": verified}, indent=2) + "\n")
    for model in args.models:
        if model == "bf16":
            continue
        expected = MODELS[model]
        metadata = get_hf_file_metadata(hf_hub_url(QUANT_REPO, expected["file"], revision=QUANT_REV))
        assert metadata.size == expected["size"] and metadata.etag == expected["sha256"]
        path = Path(hf_hub_download(QUANT_REPO, expected["file"], revision=QUANT_REV, local_dir=root / model))
        assert path.stat().st_size == expected["size"] and sha(path) == expected["sha256"]
        (path.parent / "VERIFIED.json").write_text(json.dumps({"repo": QUANT_REPO, "revision": QUANT_REV, **expected, "full_file_hash_match": True}, indent=2) + "\n")
        print(f"Verified {model}", flush=True)


if __name__ == "__main__":
    main()
