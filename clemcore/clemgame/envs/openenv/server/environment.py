import logging
import threading
from typing import Callable, Dict, Any, Optional

from openenv.core import Environment

from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.models import ClemGameState, ClemGameObservation, ClemGameAction
from clemcore.clemgame.envs.openenv.server.task_provider import ClemGameTaskProvider
from clemcore.clemgame.envs.pettingzoo import gym_env
from clemcore.clemgame.instances import GameInstances
from clemcore.clemgame.master import GameState

module_logger = logging.getLogger(__name__)

# Serializes GameBenchmark.load_from_spec() and close() across concurrent session threads;
# those calls manipulate sys.modules / sys.path and are not re-entrant — see ADR-0001.
_game_load_lock = threading.Lock()


class ClemGameEnvironment(ClemGameTaskProvider, Environment):
    # Required for max_concurrent_envs > 1; each session gets its own instance via
    # the factory in app.py, so there is no shared mutable state across sessions — see ADR-0001.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self,
                 game_name: str,
                 *,
                 instances_filter=None,
                 single_pass: bool = False,
                 learner_agent: str = "player_0",
                 env_agents: Dict[str, Any] = None,
                 game_instances: Optional[GameInstances] = None,
                 callbacks: GameBenchmarkCallbackList = None,
                 reward_func: Callable[[dict, str, GameState, dict], float] = None,
                 feedback_func: Callable[[dict, str, GameState, dict], str | None] = None
                 ):
        super().__init__()
        module_logger.info("Initialize ClemGameEnvironment: "
                           "game_name=%s, learner_agent=%s, env_agents=%s",
                           game_name, learner_agent, env_agents)
        self._game_name = game_name
        self._state = ClemGameState(game_name=game_name, episode_id="episode_0", step_count=0, episode_count=0)
        self._game_instances = game_instances
        with _game_load_lock:
            self._game_env = gym_env(game_name,
                                     instances_filter=instances_filter,
                                     single_pass=single_pass,
                                     learner_agent=learner_agent,
                                     env_agents=env_agents or {},
                                     callbacks=callbacks,
                                     reward_func=reward_func,
                                     feedback_func=feedback_func
                                     )

    def close(self):
        module_logger.info(f"Close ClemGameEnvironment {self._state.game_name}")
        with _game_load_lock:
            self._game_env.close()

    def reset(self, seed=None, episode_id=None, **kwargs) -> ClemGameObservation:
        # Translate OpenEnv standard split/index kwargs to clem's experiment/game_id format
        split = kwargs.pop("split", None)
        index = kwargs.pop("index", None)
        if split is not None:
            kwargs["experiment"] = {"name": split}
        if index is not None:
            kwargs["game_id"] = index
        if episode_id is not None:
            kwargs["episode_id"] = episode_id
        options = kwargs if kwargs else None
        module_logger.info(
            f"Reset ClemGameEnvironment '{self._state.game_name}' for episode '{self._state.episode_id}' "
            f"with kwargs={kwargs}"
        )
        observation, info = self._game_env.reset(seed=seed, options=options)
        self._state.step_count = 0
        self._state.episode_count += 1
        self._state.episode_id = f"episode_{self._state.episode_count}"
        return ClemGameObservation(context=observation)

    def step(self, action: ClemGameAction, timeout_s=None, **kwargs) -> ClemGameObservation:
        if module_logger.isEnabledFor(logging.DEBUG):
            module_logger.debug(f"Step ClemGameEnvironment with action={action.model_dump()}")
        observation, reward, done, truncated, info = self._game_env.step(action.response)
        if module_logger.isEnabledFor(logging.DEBUG):
            module_logger.debug(f"Step ClemGameEnvironment result is "
                                f"observation={observation}, reward={reward}, done={done}, "
                                f"truncated={truncated}, info={info}")
        self._state.step_count += 1
        return ClemGameObservation(
            context=observation,
            reward=float(reward),
            done=done,
            metadata={**info, **dict(truncated=truncated)}
        )

    @property
    def state(self) -> ClemGameState:
        if module_logger.isEnabledFor(logging.DEBUG):
            module_logger.debug(f"State for ClemGameEnvironment is {self._state}")
        return self._state
