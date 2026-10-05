#!/usr/bin/env python3
"""Pinned llama.cpp Metal build plus minimal all-token hidden-state C ABI."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clef_snake.catalog import NATIVE_REV


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("build"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    work = args.work.expanduser().resolve()
    source = work / "llama.cpp"
    if not source.exists():
        work.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/ggml-org/llama.cpp.git", source)
        run("git", "-C", source, "checkout", "--detach", NATIVE_REV)
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == NATIVE_REV
    patch = project / "native/llama-patch.diff"
    reverse = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], capture_output=True)
    if reverse.returncode:
        run("git", "-C", source, "apply", "--check", patch)
        run("git", "-C", source, "apply", patch)
    run("cmake", "-S", source, "-B", work / "native", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF")
    run("cmake", "--build", work / "native", "--target", "llama", "-j", "6")
    run("c++", "-std=c++17", "-dynamiclib", project / "native/bridge.cpp", "-I" + str(source / "include"),
        "-I" + str(source / "ggml/include"), "-L" + str(work / "native/bin"), "-lllama", "-lggml", "-lggml-base",
        "-Wl,-rpath," + str(work / "native/bin"), "-o", work / "libclef_bridge.dylib")
    run(sys.executable, "-m", "pip", "install", "--no-deps", source / "gguf-py")
    print(work / "libclef_bridge.dylib")


if __name__ == "__main__":
    main()
