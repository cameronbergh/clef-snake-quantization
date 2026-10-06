"""Offline path-resolution checks; no weights, model packages or network."""
from pathlib import Path
import shutil
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.kev4b_v1.local_resolver import (
    BASE_REPO, BASE_REVISION, KEV_REPO, KEV_REVISION, PINS,
    PinnedLocalResolver, SnapshotResolutionError, local_snapshot_resolution,
)


class KevLocalResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / "hub"
        self.snapshots = {}
        for repo_id, revision in PINS.items():
            path = self.cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            self.snapshots[repo_id] = path
        self.original = Mock(side_effect=AssertionError("Original resolver/network fallback called"))
        self.module = SimpleNamespace(resolve_run=self.original, marker=object())

    def resolver(self):
        return PinnedLocalResolver(self.snapshots, cache_root=self.cache)

    def test_exact_pins_and_local_paths_resolve_without_network_or_original(self):
        # Even an accidental network call fails immediately in this test.
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
                patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")):
            marker = self.module.marker
            with local_snapshot_resolution(self.module, self.snapshots, cache_root=self.cache) as resolver:
                self.assertIs(self.module.resolve_run, resolver)
                self.assertEqual(resolver(BASE_REPO + "@" + BASE_REVISION), str(self.snapshots[BASE_REPO]))
                for path in self.snapshots.values():
                    self.assertEqual(resolver(path), str(path))
                    self.assertEqual(resolver(str(path)), str(path))
                self.assertIs(self.module.marker, marker)
            self.assertIs(self.module.resolve_run, self.original)
        self.original.assert_not_called()

    def test_unknown_repos_revisions_aliases_and_paths_are_refused(self):
        resolver = self.resolver()
        invalid = [BASE_REPO, BASE_REPO + "@main", BASE_REPO + "@" + "0" * 40,
                   KEV_REPO + "@" + KEV_REVISION, "kev-latest", "other/model@" + BASE_REVISION,
                   self.root, str(self.snapshots[BASE_REPO]) + "/config.json",
                   "https://huggingface.co/" + BASE_REPO, "../snapshots", b"bytes", None]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(SnapshotResolutionError):
                resolver(value)

    def test_missing_extra_or_swapped_mapping_rejected_before_patch(self):
        variants = [
            {BASE_REPO: self.snapshots[BASE_REPO]},
            {**self.snapshots, "other/model": self.snapshots[BASE_REPO]},
            {BASE_REPO: self.snapshots[KEV_REPO], KEV_REPO: self.snapshots[BASE_REPO]},
        ]
        for mapping in variants:
            with self.subTest(keys=list(mapping)), self.assertRaises(SnapshotResolutionError):
                with local_snapshot_resolution(self.module, mapping, cache_root=self.cache):
                    self.fail("Invalid mapping installed")
            self.assertIs(self.module.resolve_run, self.original)

    def test_missing_snapshot_is_refused(self):
        shutil.rmtree(self.snapshots[BASE_REPO])
        with self.assertRaises(SnapshotResolutionError):
            self.resolver()

    def test_snapshot_removed_after_installation_is_refused_without_fallback(self):
        with local_snapshot_resolution(self.module, self.snapshots, cache_root=self.cache) as resolver:
            shutil.rmtree(self.snapshots[BASE_REPO])
            with self.assertRaises(SnapshotResolutionError):
                resolver(BASE_REPO + "@" + BASE_REVISION)
        self.original.assert_not_called()
        self.assertIs(self.module.resolve_run, self.original)

    def test_wrong_cache_root_is_refused(self):
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(SnapshotResolutionError):
            PinnedLocalResolver(self.snapshots, cache_root=other)

    def test_symlink_snapshot_directory_is_refused(self):
        path = self.snapshots[BASE_REPO]
        replacement = self.root / "replacement"
        path.rename(replacement)
        path.symlink_to(replacement, target_is_directory=True)
        with self.assertRaises(SnapshotResolutionError):
            self.resolver()

    def test_symlink_ancestor_is_refused(self):
        alias = self.root / "alias"
        alias.symlink_to(self.cache, target_is_directory=True)
        mapping = {repo: alias / path.relative_to(self.cache) for repo, path in self.snapshots.items()}
        with self.assertRaises(SnapshotResolutionError):
            PinnedLocalResolver(mapping, cache_root=alias)

    def test_symlink_file_inside_snapshot_is_refused_even_if_target_is_local(self):
        path = self.snapshots[BASE_REPO]
        (path / "linked.json").symlink_to(path / "config.json")
        with self.assertRaises(SnapshotResolutionError):
            self.resolver()

    def test_symlink_added_after_installation_is_refused(self):
        resolver = self.resolver()
        path = self.snapshots[BASE_REPO]
        (path / "linked.json").symlink_to(path / "config.json")
        with self.assertRaises(SnapshotResolutionError):
            resolver(BASE_REPO + "@" + BASE_REVISION)

    def test_context_restores_original_after_loader_exception(self):
        class LoaderFailure(Exception):
            pass
        with self.assertRaises(LoaderFailure):
            with local_snapshot_resolution(self.module, self.snapshots, cache_root=self.cache):
                self.assertEqual(self.module.resolve_run(BASE_REPO + "@" + BASE_REVISION),
                                 str(self.snapshots[BASE_REPO]))
                raise LoaderFailure("Simulated local load failure")
        self.assertIs(self.module.resolve_run, self.original)
        self.original.assert_not_called()

    def test_context_restores_original_after_base_exception(self):
        with self.assertRaises(KeyboardInterrupt):
            with local_snapshot_resolution(self.module, self.snapshots, cache_root=self.cache):
                raise KeyboardInterrupt()
        self.assertIs(self.module.resolve_run, self.original)

    def test_caller_mutation_cannot_expand_mapping_and_record_is_independent(self):
        resolver = self.resolver()
        self.snapshots["other/model"] = self.root
        record = resolver.to_record()
        record["snapshots"][BASE_REPO]["path"] = str(self.root)
        self.assertEqual(set(resolver.to_record()["snapshots"]), set(PINS))
        self.assertNotEqual(resolver.to_record()["snapshots"][BASE_REPO]["path"], str(self.root))
        with self.assertRaises(SnapshotResolutionError):
            resolver(self.root)


if __name__ == "__main__":
    unittest.main()
