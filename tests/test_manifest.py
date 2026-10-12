import pathlib
import unittest

from openenv.validation.parsers.openenv_yaml import OpenEnvYamlParser

SERVER_DIR = pathlib.Path(__file__).parent.parent / "clemcore/clemgame/envs/openenv/server"


class ManifestValidationTestCase(unittest.TestCase):
    """openenv.yaml must parse and validate against the OpenEnv manifest schema."""

    def setUp(self):
        self.parser = OpenEnvYamlParser()

    def test_manifest_file_exists(self):
        self.assertTrue((SERVER_DIR / "openenv.yaml").exists())

    def test_manifest_parses_without_error(self):
        manifest = self.parser.parse(SERVER_DIR)
        self.assertIsNotNone(manifest)

    def test_manifest_name_is_clem_env(self):
        manifest = self.parser.parse(SERVER_DIR)
        self.assertEqual(manifest.name, "clem_env")

    def test_manifest_reward_range_covers_aborted(self):
        manifest = self.parser.parse(SERVER_DIR)
        low, high = manifest.reward.range
        self.assertLessEqual(low, -1.0)
        self.assertGreaterEqual(high, 1.0)

    def test_manifest_declares_task_api(self):
        manifest = self.parser.parse(SERVER_DIR)
        self.assertTrue(manifest.capabilities.task_api)


if __name__ == "__main__":
    unittest.main()
