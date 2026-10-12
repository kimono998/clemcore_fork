import unittest

from clemcore.clemgame.instances import GameInstances
from clemcore.clemgame.envs.openenv.server.task_provider import ClemGameTaskProvider

FAKE_ROWS = [
    {"game_name": "g", "experiment": {"name": "train"}, "game_instance": {"game_id": 0}},
    {"game_name": "g", "experiment": {"name": "train"}, "game_instance": {"game_id": 1}},
    {"game_name": "g", "experiment": {"name": "train"}, "game_instance": {"game_id": 2}},
    {"game_name": "g", "experiment": {"name": "test"}, "game_instance": {"game_id": 0}},
    {"game_name": "g", "experiment": {"name": "test"}, "game_instance": {"game_id": 1}},
]


class StandaloneProvider(ClemGameTaskProvider):
    """Minimal concrete class for unit-testing the mixin in isolation."""

    def __init__(self, game_instances):
        self._game_instances = game_instances


class ListSplitsTestCase(unittest.TestCase):

    def setUp(self):
        self.provider = StandaloneProvider(GameInstances("g", FAKE_ROWS))

    def test_returns_all_experiment_names(self):
        self.assertIn("train", self.provider.list_splits())
        self.assertIn("test", self.provider.list_splits())

    def test_returns_sorted_names(self):
        self.assertEqual(self.provider.list_splits(), ["test", "train"])

    def test_no_duplicates(self):
        self.assertEqual(len(self.provider.list_splits()), 2)

    def test_empty_when_no_instances(self):
        self.assertEqual(StandaloneProvider(None).list_splits(), [])


class NumTasksTestCase(unittest.TestCase):

    def setUp(self):
        self.provider = StandaloneProvider(GameInstances("g", FAKE_ROWS))

    def test_counts_instances_in_train(self):
        self.assertEqual(self.provider.num_tasks("train"), 3)

    def test_counts_instances_in_test(self):
        self.assertEqual(self.provider.num_tasks("test"), 2)

    def test_unknown_split_returns_zero(self):
        self.assertEqual(self.provider.num_tasks("no_such_split"), 0)

    def test_zero_when_no_instances(self):
        self.assertEqual(StandaloneProvider(None).num_tasks("train"), 0)


class GetTaskTestCase(unittest.TestCase):

    def setUp(self):
        self.provider = StandaloneProvider(GameInstances("g", FAKE_ROWS))

    def test_returns_experiment_and_game_id(self):
        task = self.provider.get_task("train", 0)
        self.assertEqual(task, {"experiment": {"name": "train"}, "game_id": 0})

    def test_last_valid_index(self):
        task = self.provider.get_task("train", 2)
        self.assertEqual(task["game_id"], 2)

    def test_index_equals_game_id(self):
        for i in range(3):
            task = self.provider.get_task("train", i)
            self.assertEqual(task["game_id"], i)

    def test_raises_index_error_on_negative(self):
        with self.assertRaises(IndexError):
            self.provider.get_task("train", -1)

    def test_raises_index_error_on_out_of_range(self):
        with self.assertRaises(IndexError):
            self.provider.get_task("train", 3)

    def test_raises_index_error_for_unknown_split(self):
        with self.assertRaises(IndexError):
            self.provider.get_task("no_such_split", 0)


class GetTaskRangeTestCase(unittest.TestCase):

    def setUp(self):
        self.provider = StandaloneProvider(GameInstances("g", FAKE_ROWS))

    def test_full_range_equals_list_tasks(self):
        self.assertEqual(
            self.provider.get_task_range("train"),
            self.provider.list_tasks("train")
        )

    def test_start_and_stop(self):
        tasks = self.provider.get_task_range("train", start=1, stop=3)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["game_id"], 1)
        self.assertEqual(tasks[1]["game_id"], 2)

    def test_stop_only(self):
        tasks = self.provider.get_task_range("train", stop=2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["game_id"], 0)

    def test_start_only(self):
        tasks = self.provider.get_task_range("train", start=2)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["game_id"], 2)

    def test_empty_range(self):
        self.assertEqual(self.provider.get_task_range("train", start=2, stop=2), [])

    def test_unknown_split_returns_empty(self):
        self.assertEqual(self.provider.get_task_range("no_such_split"), [])


class ListTasksTestCase(unittest.TestCase):

    def setUp(self):
        self.provider = StandaloneProvider(GameInstances("g", FAKE_ROWS))

    def test_returns_all_tasks_for_split(self):
        tasks = self.provider.list_tasks("train")
        self.assertEqual(len(tasks), 3)

    def test_each_task_has_experiment_and_game_id(self):
        for i, task in enumerate(self.provider.list_tasks("train")):
            self.assertEqual(task["experiment"]["name"], "train")
            self.assertEqual(task["game_id"], i)


if __name__ == "__main__":
    unittest.main()
