"""Train a PPO agent on BalatroEnv, then compare it against the baselines.

Usage: python scripts/train_ppo.py [total_timesteps]
"""
from __future__ import annotations

import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from balatro_sim.agents import heuristic_action, random_action, run_episode
from balatro_sim.env import BalatroEnv

CHECKPOINT_PATH = "checkpoints/ppo_balatro"
EVAL_EPISODES = 30


def ppo_action(model):
    def agent_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return agent_fn


def summarize(name: str, results: list[dict]) -> None:
    import statistics
    rewards = [r["reward"] for r in results]
    antes = [r["ante"] for r in results]
    wins = sum(1 for r in results if r["result"] == "won_run")
    print(f"{name}: mean_reward={statistics.mean(rewards):.3f} mean_ante={statistics.mean(antes):.2f} wins={wins}/{len(results)}")


def main() -> None:
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000

    vec_env = DummyVecEnv([BalatroEnv])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(CHECKPOINT_PATH)

    ppo_results = [run_episode(ppo_action(model), seed=i) for i in range(EVAL_EPISODES)]
    random_results = [run_episode(random_action, seed=i) for i in range(EVAL_EPISODES)]
    heuristic_results = [run_episode(heuristic_action, seed=i) for i in range(EVAL_EPISODES)]

    print()
    summarize("random", random_results)
    summarize("heuristic", heuristic_results)
    summarize("ppo", ppo_results)


if __name__ == "__main__":
    main()
