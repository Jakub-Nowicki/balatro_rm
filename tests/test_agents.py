import numpy as np

from balatro_sim.agents import heuristic_action, random_action, run_episode
from balatro_sim.env import HAND_SLOTS, BalatroEnv


def test_random_action_has_correct_shape():
    env = BalatroEnv()
    env.reset(seed=0)
    action = random_action(env)
    assert action.shape == (HAND_SLOTS + 2,)


def test_heuristic_action_is_always_valid_across_seeds():
    for seed in range(10):
        env = BalatroEnv()
        env.reset(seed=seed)
        action = heuristic_action(env)
        n_selected = sum(action[:HAND_SLOTS])
        mode = action[HAND_SLOTS]
        assert 1 <= n_selected <= 5
        assert mode in (0, 1)


def test_heuristic_outperforms_random_on_average():
    random_rewards = [run_episode(random_action, s)["reward"] for s in range(15)]
    heuristic_rewards = [run_episode(heuristic_action, s)["reward"] for s in range(15)]

    assert np.mean(heuristic_rewards) > np.mean(random_rewards)
