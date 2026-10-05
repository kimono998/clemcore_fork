import pytest
from clemcore.clemgame.instances import GameInstances

def make_instances():
    rows = [
        {"game_name": "g", "experiment": {"name": "easy"}, "game_instance": {"game_id": 1, "word": "cat"}},
        {"game_name": "g", "experiment": {"name": "hard"}, "game_instance": {"game_id": 1, "word": "dog"}},
    ]
    return GameInstances("g", rows)


def test_find_by_game_id_respects_experiment():
    row = make_instances().find_by_game_id(1, "hard")
    assert row["game_instance"]["word"] == "dog"


def test_find_by_game_id_same_id_other_experiment():
    row = make_instances().find_by_game_id(1, "easy")
    assert row["game_instance"]["word"] == "cat"


def test_find_by_game_id_requires_experiment():
    with pytest.raises(TypeError):
        make_instances().find_by_game_id(1)


def test_find_by_game_id_not_found_in_experiment():
    with pytest.raises(ValueError):
        make_instances().find_by_game_id(99, "easy")


def test_find_by_game_id_unknown_experiment():
    with pytest.raises(ValueError):
        make_instances().find_by_game_id(1, "medium")
