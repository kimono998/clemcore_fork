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
    def _task_dict(split: str, game_id) -> dict:
        return {"experiment": {"name": split}, "game_id": game_id}

    def _rows_for_split(self, split: str) -> list:
        if not self._game_instances:
            return []
        return [r for r in self._game_instances if r["experiment"]["name"] == split]

    def list_splits(self) -> list:
        if not self._game_instances:
            return []
        return self._game_instances.experiment_names

    def list_tasks(self, split: str) -> list:
        return [self._task_dict(split, r["game_instance"]["game_id"])
                for r in self._rows_for_split(split)]

    def num_tasks(self, split: str) -> int:
        return self._game_instances.count_by_experiment(split) if self._game_instances else 0

    def get_task(self, split: str, index: int) -> dict:
        rows = self._rows_for_split(split)
        if not (0 <= index < len(rows)):
            raise IndexError(
                f"task index {index} out of range for split {split!r} with {len(rows)} tasks"
            )
        return self._task_dict(split, rows[index]["game_instance"]["game_id"])

    def get_task_range(self, split: str, start: int = None, stop: int = None) -> list:
        rows = self._rows_for_split(split)
        return [self._task_dict(split, rows[i]["game_instance"]["game_id"])
                for i in range(*slice(start, stop).indices(len(rows)))]
