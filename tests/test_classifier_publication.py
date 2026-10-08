import unittest
from experiments.classifier_benchmark_publication_v1.export import inspect_public_payload,PUBLIC_AUTH_EXAMPLE
class PublicationPrivacy(unittest.TestCase):
    def test_exact_hash_verified_public_example_preserved(self):self.assertEqual(inspect_public_payload(PUBLIC_AUTH_EXAMPLE.encode()),1)
    def test_other_bearer_marker_rejected(self):
        with self.assertRaisesRegex(AssertionError,'private content marker'):inspect_public_payload(b'Authorization: Bearer real-private-value')
    def test_appended_private_value_is_not_exempted(self):
        with self.assertRaisesRegex(AssertionError,'private content marker'):inspect_public_payload((PUBLIC_AUTH_EXAMPLE+' Authorization: Bearer real-value').encode())
    def test_private_paths_rejected(self):
        for p in [b'/Users/private-person/models',b'/home/private-person/token']:
            with self.assertRaisesRegex(AssertionError,'private content marker'):inspect_public_payload(p)
