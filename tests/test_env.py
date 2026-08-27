import numpy as np

from balatro_sim.env import HAND_SLOTS, OBS_SIZE, BalatroEnv


def test_reset_returns_correctly_shaped_obs():
    env = BalatroEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert info == {}


def test_action_space_and_observation_space_shapes():
    env = BalatroEnv()
    assert env.action_space.shape == (HAND_SLOTS + 1,)
    assert env.observation_space.shape == (OBS_SIZE,)


def test_invalid_action_penalized_without_crashing():
    env = BalatroEnv()
    env.reset(seed=0)
    no_cards_selected = np.zeros(HAND_SLOTS + 1, dtype=int)  # 0 cards selected, mode=play
    obs, reward, terminated, truncated, info = env.step(no_cards_selected)
    assert reward < 0
    assert not terminated
    assert not truncated


def test_valid_play_scores_and_advances_state():
    env = BalatroEnv()
    env.reset(seed=0)
    action = np.zeros(HAND_SLOTS + 1, dtype=int)
    action[0] = 1  # select first hand slot
    action[HAND_SLOTS] = 0  # play mode
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.game.hands_remaining == env.game.hands_per_round - 1


def test_episode_terminates_with_random_actions():
    env = BalatroEnv()
    env.reset(seed=42)
    rng = np.random.RandomState(42)
    for _ in range(200):
        action = rng.randint(0, 2, size=HAND_SLOTS + 1)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert info.get("result") in ("lost", "won_run")
            return
    raise AssertionError("episode did not terminate within 200 random steps")
