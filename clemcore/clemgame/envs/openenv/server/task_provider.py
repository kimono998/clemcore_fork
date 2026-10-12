from typing import Optional

from clemcore.clemgame.instances import GameInstances


class ClemGameTaskProvider:
    """Mixin that adds OpenEnv TaskProvider discovery methods to ClemGameEnvironment.

    Maps clem domain concepts to OpenEnv's task API:
      - split  = experiment name
      - task   = game instance (identified by game_id)
      - index  = zero-based position within the experiment's instances

    Requires subclasses to set self._game_instances before calling any method.
    When _game_instances is None, all methods return empty results.
    """

    _game_instances: Optional[GameInstances]

    @staticmethod
    def _task_dict(split: str, index: int) -> dict:
        return {"experiment": {"name": split}, "game_id": index}

    def list_splits(self) -> list:
        if not self._game_instances:
            return []
        return self._game_instances.experiment_names

    def list_tasks(self, split: str) -> list:
        return [self._task_dict(split, i) for i in range(self.num_tasks(split))]

    def num_tasks(self, split: str) -> int:
        if not self._game_instances:
            return 0
        return self._game_instances.count_by_experiment(split)

    def get_task(self, split: str, index: int) -> dict:
        n = self.num_tasks(split)
        if not (0 <= index < n):
            raise IndexError(
                f"task index {index} out of range for split {split!r} with {n} tasks"
            )
        return self._task_dict(split, index)

    def get_task_range(self, split: str, start: int = None, stop: int = None) -> list:
        n = self.num_tasks(split)
        return [self._task_dict(split, i) for i in range(*slice(start, stop).indices(n))]
