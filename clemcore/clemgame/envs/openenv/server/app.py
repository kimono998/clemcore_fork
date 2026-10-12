import itertools
import os
from typing import Callable, Dict, Optional, Any

from datasets import load_dataset
from openenv.core import create_app

from clemcore.backends import load_models
from clemcore.utils.string_utils import read_query_string
from clemcore.clemgame.callbacks import episode_results_folder_callbacks
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation, ClemGameState
from clemcore.clemgame.envs.openenv.server.environment import ClemGameEnvironment
from clemcore.clemgame.envs.pettingzoo import check_agent_mapping_for_training
from clemcore.clemgame.master import GameState
from clemcore.clemgame.registry import GameRegistry
from clemcore.clemgame.instances import to_instance_filter

CLEMV_GAME = os.getenv("CLEMV_GAME")
CLEMV_GAME_SPLIT = os.getenv("CLEMV_GAME_SPLIT")
CLEMV_SINGLE_PASS = os.getenv("CLEMV_SINGLE_PASS", "False").lower() in ("true", "1", "t")
CLEMV_LEARNER_AGENT = os.getenv("CLEMV_LEARNER_AGENT")
CLEMV_ENV_AGENTS = read_query_string(os.getenv("CLEMV_ENV_AGENTS"))
CLEMV_GEN_ARGS = read_query_string(os.getenv("CLEMV_GEN_ARGS"))
CLEMV_RESULTS_DIR = os.getenv("CLEMV_RESULTS_DIR", "openenv-records")
CLEMV_RUN_ID = os.getenv("CLEMV_RUN_ID")
MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))


def create_clemv_app(
        game_name: str = CLEMV_GAME,
        *,
        learner_agent: str = CLEMV_LEARNER_AGENT,
        env_agents: Optional[Dict[str, str]] = None,
        game_instance_split: str = CLEMV_GAME_SPLIT,
        single_pass: bool = CLEMV_SINGLE_PASS,
        gen_args: Optional[Dict[str, Any]] = None,
        callbacks: Optional[GameBenchmarkCallbackList] = None,
        reward_func: Optional[Callable[[dict, str, GameState, dict], float]] = None,
        feedback_func: Optional[Callable[[dict, str, GameState, dict], str | None]] = None,
        results_dir: Optional[str] = None,
        run_id: Optional[str] = None
):
    env_agents = env_agents if env_agents is not None else CLEMV_ENV_AGENTS
    gen_args = gen_args if gen_args is not None else CLEMV_GEN_ARGS
    results_dir = results_dir if results_dir is not None else CLEMV_RESULTS_DIR
    run_id = run_id if run_id is not None else CLEMV_RUN_ID

    missing = [k for k, v in {"game_name": game_name, "learner_agent": learner_agent}.items() if v is None]
    if missing:
        raise ValueError(f"Missing required configuration for: {', '.join(missing)}. "
                         "Provide them as arguments or set the corresponding CLEM_GAME_* env vars.")

    # One-time expensive setup shared across all sessions
    game_spec = GameRegistry.from_directories_and_cwd_files().get_game_spec(game_name)
    check_agent_mapping_for_training(game_spec, {learner_agent: "learner", **(env_agents or {})})

    instances_filter = None
    if game_instance_split:
        dataset = load_dataset("colab-potsdam/playpen-data", "instances", split=game_instance_split)
        instances_filter = to_instance_filter(dataset)

    env_agent_models: Dict[str, Any] = {}
    if game_spec.is_multi_player() and env_agents:
        agent_models = load_models(list(env_agents.values()), gen_args)
        env_agent_models = {agent_id: model for agent_id, model in zip(env_agents.keys(), agent_models)}

    if run_id is None and env_agents:
        run_id = "-".join(env_agents.values())

    _session_counter = itertools.count()

    def make_env():
        session_idx = next(_session_counter)
        if callbacks is None and results_dir is not None:
            session_callbacks = episode_results_folder_callbacks(
                run_dir=f"{run_id or 'run'}/session_{session_idx:04d}",
                result_dir_path=results_dir
            )
        else:
            session_callbacks = callbacks
        return ClemGameEnvironment(
            game_name,
            instances_filter=instances_filter,
            single_pass=single_pass,
            learner_agent=learner_agent,
            env_agents=env_agent_models,
            callbacks=session_callbacks,
            reward_func=reward_func,
            feedback_func=feedback_func
        )

    return create_app(make_env, ClemGameAction, ClemGameObservation,
                      state_cls=ClemGameState,
                      env_name="clem_env",
                      max_concurrent_envs=MAX_CONCURRENT_ENVS)
