"""Resolve verified Kev inputs locally without importing a model or Hub client.

The caller must verify the complete pinned snapshot file hashes and storage
identity before using this boundary. This module validates paths and pins; it
does not independently establish payload integrity or authorize model loading.

The pinned upstream checkpoint loader remains unchanged. During one serial
load, ``local_snapshot_resolution`` substitutes only its path resolver. All
unknown identifiers are refused; the original resolver is never a fallback.
Encoding, LoRA merging, head weights, calibration and forward math are untouched.
"""
from contextlib import contextmanager
import os
from pathlib import Path
from types import MappingProxyType


BASE_REPO = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_REPO = "jaredpalmer/kev-4b"
KEV_REVISION = "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
PINS = MappingProxyType({BASE_REPO: BASE_REVISION, KEV_REPO: KEV_REVISION})


class SnapshotResolutionError(ValueError):
    """An input is outside the previously verified local snapshot allowlist."""


def _directory(value):
    """Require a canonical absolute directory with no symlink ancestors."""
    try:
        path = Path(value)
        if not path.is_absolute() or not path.is_dir() or path.resolve(strict=True) != path:
            raise SnapshotResolutionError("Snapshot paths must be existing canonical absolute directories")
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise SnapshotResolutionError("Symlinks are forbidden in snapshot paths")
        return path
    except (OSError, RuntimeError, TypeError) as exc:
        raise SnapshotResolutionError("Cannot validate local snapshot directory") from exc


class PinnedLocalResolver:
    """Allow exactly the pinned base identifier and two verified local paths.

    ``verified_snapshots`` maps repository IDs to already hash-verified snapshot
    directories. Its keys must be exactly ``PINS``. The explicit ``cache_root``
    is the already storage-verified Hugging Face hub directory. No directory or
    file is created, downloaded, copied, loaded or changed here.
    """

    def __init__(self, verified_snapshots, *, cache_root):
        self._cache_root = _directory(cache_root)
        if set(verified_snapshots) != set(PINS):
            raise SnapshotResolutionError("Exactly the two pinned source snapshots are required")
        snapshots = {}
        for repo_id, revision in PINS.items():
            path = _directory(verified_snapshots[repo_id])
            expected = (self._cache_root / ("models--" + repo_id.replace("/", "--"))
                        / "snapshots" / revision)
            if path != expected:
                raise SnapshotResolutionError("Snapshot path does not match its exact repository and revision")
            self._check_contents(path)
            snapshots[repo_id] = path
        # Copy mappings so later caller mutation cannot expand the allowlist.
        self._snapshots = MappingProxyType(snapshots)
        self._local_paths = MappingProxyType({str(path): path for path in snapshots.values()})

    @staticmethod
    def _check_contents(path):
        # Payload hashing is the caller's responsibility. Reject links even
        # inside the verified directory so a loader cannot follow an escape.
        try:
            if any(item.is_symlink() for item in path.rglob("*")):
                raise SnapshotResolutionError("Symlinks inside snapshots are forbidden")
        except OSError as exc:
            raise SnapshotResolutionError("Cannot inspect local snapshot paths") from exc

    def __call__(self, run):
        try:
            requested = os.fspath(run)
        except TypeError as exc:
            raise SnapshotResolutionError("A pinned identifier or verified local path is required") from exc
        if not isinstance(requested, str):
            raise SnapshotResolutionError("Snapshot identifiers must be text")
        if requested == BASE_REPO + "@" + BASE_REVISION:
            path = self._snapshots[BASE_REPO]
        elif requested in self._local_paths:
            path = self._local_paths[requested]
        else:
            raise SnapshotResolutionError("Unapproved snapshot identifier or local path")
        # Recheck before every resolution; a removed/replaced directory is a
        # technical stop, never permission for a download or alternate path.
        _directory(path)
        self._check_contents(path)
        return str(path)

    def to_record(self):
        """Return an independent JSON-safe description for the caller's log."""
        return {
            "kind": "pinned_local_snapshot_resolution_v1",
            "cache_root": str(self._cache_root),
            "pinned_base_identifier": BASE_REPO + "@" + BASE_REVISION,
            "snapshots": {repo_id: {"revision": PINS[repo_id], "path": str(path)}
                          for repo_id, path in self._snapshots.items()},
            "network_calls": 0,
            "unknown_input_policy": "refuse_without_fallback",
            "payload_hash_verification": "required_from_caller_before_installation",
        }


@contextmanager
def local_snapshot_resolution(checkpoint_module, verified_snapshots, *, cache_root):
    """Temporarily replace one module's resolver, restoring it on every exit.

    Pass the already imported, source-verified upstream ``kev.checkpoint``
    module. No upstream import occurs here. This scope is for a single serial
    loader; callers must not share it with concurrent loading threads.
    """
    resolver = PinnedLocalResolver(verified_snapshots, cache_root=cache_root)
    original = checkpoint_module.resolve_run
    if not callable(original):
        raise SnapshotResolutionError("Checkpoint module must expose a callable resolve_run")
    checkpoint_module.resolve_run = resolver
    try:
        yield resolver
    finally:
        checkpoint_module.resolve_run = original
