import socket
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import httpx
import pytest
import uvicorn

from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.client import ClemGameEnv
from clemcore.clemgame.envs.openenv.models import ClemGameAction
from clemcore.clemgame.envs.openenv.server.app import create_clemv_app
from clemcore.clemgame.envs.openenv.server.environment import ClemGameEnvironment
from clemcore.clemgame.instances import GameInstances

APP_MODULE = "clemcore.clemgame.envs.openenv.server.app"
ENVIRONMENT_MODULE = "clemcore.clemgame.envs.openenv.server.environment"

FAKE_PROVIDER_ROWS = [
    {"game_name": "fake_game", "experiment": {"name": "high_frequency_words"}, "game_instance": {"game_id": 0}},
    {"game_name": "fake_game", "experiment": {"name": "high_frequency_words"}, "game_instance": {"game_id": 1}},
    {"game_name": "fake_game", "experiment": {"name": "high_frequency_words"}, "game_instance": {"game_id": 2}},
    {"game_name": "fake_game", "experiment": {"name": "low_frequency_words"}, "game_instance": {"game_id": 0}},
]


class FakeGameEnv:
    """Stands in for gym_env: echoes actions and ends the episode on 'stop'."""

    def __init__(self):
        self.reset_calls = []

    def reset(self, seed=None, options=None):
        self.reset_calls.append(dict(seed=seed, options=options))
        return {"role": "user", "content": "start"}, {}

    def step(self, action):
        done = action == "stop"
        observation = {"role": "user", "content": f"echo {action}"}
        return observation, 1.0 if done else 0.0, done, False, {"turn_feedback": "feedback"}

    def close(self):
        pass


def patch_game_env(game_instances=None, multiplayer=False):
    """Replaces game loading with FakeGameEnv; set multiplayer=True to also mock load_models."""
    game_name = "fake_game_2p" if multiplayer else "fake_game"
    game_spec = MagicMock()
    game_spec.is_multi_player.return_value = multiplayer
    registry = MagicMock()
    registry.from_directories_and_cwd_files.return_value.get_game_spec.return_value = game_spec
    created = []

    def make_fake_env(*args, **kwargs):
        created.append(FakeGameEnv())
        return created[-1]

    instances_cls_mock = MagicMock()
    instances_cls_mock.from_game_spec.return_value = (
        game_instances if game_instances is not None else GameInstances(game_name, [])
    )

    patches = [
        patch(f"{APP_MODULE}.GameRegistry", registry),
        patch(f"{APP_MODULE}.check_agent_mapping_for_training"),
        patch(f"{APP_MODULE}.GameInstances", instances_cls_mock),
        patch(f"{ENVIRONMENT_MODULE}.gym_env", side_effect=make_fake_env),
    ]
    if multiplayer:
        patches.append(patch(f"{APP_MODULE}.load_models", return_value=[MagicMock()]))
    return patches, created


def free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class ServerThread:
    """Runs an app with uvicorn in a background thread."""

    def __init__(self, app):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.05)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def client(self):
        return ClemGameEnv(base_url=f"http://localhost:{self.port}").sync()


class ClemGameEnvironmentTestCase(unittest.TestCase):
    """Tests the OpenEnv Environment adapter around gym_env, without a server."""

    def setUp(self):
        self.patches, self.created = patch_game_env()
        for p in self.patches:
            p.start()
        self.env = ClemGameEnvironment("fake_game")
        self.game_env = self.created[-1]

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_reset_forwards_kwargs_as_options(self):
        self.env.reset(experiment={"name": "exp1"}, game_id=1)
        self.assertEqual(self.game_env.reset_calls[-1]["options"], {"experiment": {"name": "exp1"}, "game_id": 1})

    def test_reset_without_kwargs_passes_no_options(self):
        self.env.reset()
        self.assertIsNone(self.game_env.reset_calls[-1]["options"])

    def test_reset_forwards_episode_id_as_option(self):
        self.env.reset(episode_id="my_episode")
        self.assertEqual(self.game_env.reset_calls[-1]["options"], {"episode_id": "my_episode"})

    def test_reset_returns_context(self):
        observation = self.env.reset()
        self.assertEqual(observation.context, {"role": "user", "content": "start"})

    def test_step_maps_gym_result(self):
        self.env.reset()
        observation = self.env.step(ClemGameAction(response="stop"))
        self.assertEqual(observation.context["content"], "echo stop")
        self.assertEqual(observation.reward, 1.0)
        self.assertTrue(observation.done)
        self.assertEqual(observation.metadata, {"turn_feedback": "feedback", "truncated": False})

    def test_state_counts_steps_and_episodes(self):
        self.env.reset()
        self.env.step(ClemGameAction(response="hello"))
        self.assertEqual(self.env.state.step_count, 1)
        self.assertEqual(self.env.state.episode_count, 1)
        self.env.reset()
        self.assertEqual(self.env.state.step_count, 0)
        self.assertEqual(self.env.state.episode_count, 2)
        self.assertEqual(self.env.state.game_name, "fake_game")

    def test_reset_translates_split_and_index_to_clem_format(self):
        self.env.reset(split="high_frequency_words", index=3)
        options = self.game_env.reset_calls[-1]["options"]
        self.assertEqual(options["experiment"], {"name": "high_frequency_words"})
        self.assertEqual(options["game_id"], 3)

    def test_reset_split_without_index_still_works(self):
        self.env.reset(split="train")
        options = self.game_env.reset_calls[-1]["options"]
        self.assertEqual(options["experiment"], {"name": "train"})
        self.assertNotIn("game_id", options)

    def test_reset_index_without_split_still_works(self):
        self.env.reset(index=5)
        options = self.game_env.reset_calls[-1]["options"]
        self.assertEqual(options["game_id"], 5)
        self.assertNotIn("experiment", options)

    def test_reset_resolves_index_to_actual_game_id_when_instances_available(self):
        """Positional index must map to the stored game_id, not the index itself."""
        instances = GameInstances("fake_game", [
            {"game_name": "fake_game", "experiment": {"name": "train"},
             "game_instance": {"game_id": 10}},
            {"game_name": "fake_game", "experiment": {"name": "train"},
             "game_instance": {"game_id": 20}},
        ])
        env = ClemGameEnvironment("fake_game", game_instances=instances)
        env._game_env = self.game_env  # reuse the already-created fake
        env.reset(split="train", index=0)
        self.assertEqual(self.game_env.reset_calls[-1]["options"]["game_id"], 10)
        env.reset(split="train", index=1)
        self.assertEqual(self.game_env.reset_calls[-1]["options"]["game_id"], 20)


class OpenEnvServerClientTestCase(unittest.TestCase):
    """Round trips through the real OpenEnv server and client, with a fake game."""

    @classmethod
    def setUpClass(cls):
        cls.patches, cls.created = patch_game_env()
        for p in cls.patches:
            p.start()
        app = create_clemv_app("fake_game", learner_agent="player_0", callbacks=GameBenchmarkCallbackList())
        cls.server = ServerThread(app)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        for p in cls.patches:
            p.stop()

    def test_reset_and_step(self):
        with self.server.client() as env:
            result = env.reset()
            self.assertFalse(result.done)
            self.assertEqual(result.observation.context["content"], "start")

            result = env.step(ClemGameAction(response="hello"))
            self.assertFalse(result.done)
            self.assertEqual(result.reward, 0.0)
            self.assertEqual(result.observation.context["content"], "echo hello")

            result = env.step(ClemGameAction(response="stop"))
            self.assertTrue(result.done)
            self.assertEqual(result.reward, 1.0)

    def test_reset_options_reach_game_env(self):
        with self.server.client() as env:
            env.reset(experiment={"name": "exp1"}, game_id=1)
        options = self.created[-1].reset_calls[-1]["options"]
        self.assertEqual(options["experiment"], {"name": "exp1"})
        self.assertEqual(options["game_id"], 1)

    def test_metadata_reaches_client(self):
        with self.server.client() as env:
            env.reset()
            result = env.step(ClemGameAction(response="hello"))
        self.assertEqual(result.observation.metadata, {"turn_feedback": "feedback", "truncated": False})

    def test_step_result_metadata_is_populated(self):
        with self.server.client() as env:
            env.reset()
            result = env.step(ClemGameAction(response="hello"))
        self.assertIsNotNone(result.metadata)
        self.assertEqual(result.metadata, {"turn_feedback": "feedback", "truncated": False})

    def test_state_reaches_client(self):
        with self.server.client() as env:
            env.reset()
            state = env.state()
        self.assertEqual(state.game_name, "fake_game")
        self.assertEqual(state.episode_count, 1)


class TaskProviderServerTestCase(unittest.TestCase):
    """HTTP task-discovery routes return data from the game instances."""

    @classmethod
    def setUpClass(cls):
        fake_instances = GameInstances("fake_game", FAKE_PROVIDER_ROWS)
        cls.patches, cls.created = patch_game_env(game_instances=fake_instances)
        for p in cls.patches:
            p.start()
        app = create_clemv_app("fake_game", learner_agent="player_0", callbacks=GameBenchmarkCallbackList())
        cls.server = ServerThread(app)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        for p in cls.patches:
            p.stop()

    def test_splits_endpoint_returns_experiment_names(self):
        resp = httpx.get(f"http://localhost:{self.server.port}/clem_env/splits")
        self.assertEqual(resp.status_code, 200)
        names = [s["name"] for s in resp.json()]
        self.assertIn("high_frequency_words", names)
        self.assertIn("low_frequency_words", names)

    def test_num_tasks_endpoint_returns_instance_count(self):
        resp = httpx.post(
            f"http://localhost:{self.server.port}/clem_env/num_tasks",
            json={"split": "high_frequency_words"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["num_tasks"], 3)

    def test_get_task_endpoint_returns_experiment_and_game_id(self):
        resp = httpx.post(
            f"http://localhost:{self.server.port}/clem_env/task",
            json={"split": "high_frequency_words", "index": 1}
        )
        self.assertEqual(resp.status_code, 200)
        task = resp.json()["task"]
        self.assertEqual(task["experiment"]["name"], "high_frequency_words")
        self.assertEqual(task["game_id"], 1)


def _run_two_concurrent_sessions(server):
    """Asserts two clients can hold sessions simultaneously; returns (results, errors)."""
    barrier = threading.Barrier(2)
    results = {}
    errors = []

    def run_session(key):
        try:
            with server.client() as env:
                env.reset()
                barrier.wait(timeout=5)
                result = env.step(ClemGameAction(response="hello"))
                results[key] = result.observation.context["content"]
        except Exception as e:
            errors.append((key, e))

    t1 = threading.Thread(target=run_session, args=("a",))
    t2 = threading.Thread(target=run_session, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    return results, errors


class ConcurrentSessionsTestCase(unittest.TestCase):
    """Proves that two 1-player sessions run simultaneously without blocking or corrupting each other."""

    @classmethod
    def setUpClass(cls):
        cls.patches, cls.created = patch_game_env()
        for p in cls.patches:
            p.start()
        app = create_clemv_app("fake_game", learner_agent="player_0", callbacks=GameBenchmarkCallbackList())
        cls.server = ServerThread(app)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        for p in cls.patches:
            p.stop()

    def test_two_sessions_run_concurrently(self):
        """Two clients must hold sessions simultaneously (SUPPORTS_CONCURRENT_SESSIONS + max_concurrent_envs > 1)."""
        results, errors = _run_two_concurrent_sessions(self.server)
        self.assertFalse(errors, f"Session errors: {errors}")
        self.assertEqual(results.get("a"), "echo hello")
        self.assertEqual(results.get("b"), "echo hello")


class Concurrent2PlayerSessionsTestCase(unittest.TestCase):
    """Proves that two 2-player sessions run simultaneously without blocking or corrupting each other."""

    @classmethod
    def setUpClass(cls):
        cls.patches, cls.created = patch_game_env(multiplayer=True)
        for p in cls.patches:
            p.start()
        app = create_clemv_app(
            "fake_game_2p",
            learner_agent="player_0",
            env_agents={"player_1": "mock-model"},
            callbacks=GameBenchmarkCallbackList()
        )
        cls.server = ServerThread(app)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        for p in cls.patches:
            p.stop()

    def test_two_2player_sessions_run_concurrently(self):
        """Two clients in 2-player games must hold sessions simultaneously."""
        results, errors = _run_two_concurrent_sessions(self.server)
        self.assertFalse(errors, f"Session errors: {errors}")
        self.assertEqual(results.get("a"), "echo hello")
        self.assertEqual(results.get("b"), "echo hello")


@pytest.mark.clembench
class OpenEnvWordleTestCase(unittest.TestCase):
    """Plays the real Wordle game through the OpenEnv server and client."""

    @classmethod
    def setUpClass(cls):
        app = create_clemv_app("wordle", learner_agent="player_0", callbacks=GameBenchmarkCallbackList())
        cls.server = ServerThread(app)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_valid_guess_continues_game(self):
        with self.server.client() as env:
            env.reset(experiment={"name": "high_frequency_words_no_clue_no_critic"}, game_id=1)
            result = env.step(ClemGameAction(response="explanation: test\nguess: crane"))
        self.assertFalse(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertIn("guess_feedback", result.observation.context["content"])


if __name__ == '__main__':
    unittest.main()
