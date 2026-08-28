"""Run random vs. heuristic agents through BalatroEnv and print summary stats.

Usage: python scripts/run_baseline.py [num_episodes] [--no-shop]
"""
from __future__ import annotations

import statistics
import sys
from functools import partial

from balatro_sim.agents import heuristic_action, random_action, run_episode
from balatro_sim.env import BalatroEnv


def summarize(name: str, results: list[dict]) -> None:
    rewards = [r["reward"] for r in results]
    antes = [r["ante"] for r in results]
    wins = sum(1 for r in results if r["result"] == "won_run")
    print(f"{name}: mean_reward={statistics.mean(rewards):.3f} mean_ante={statistics.mean(antes):.2f} wins={wins}/{len(results)}")


def main() -> None:
    args = sys.argv[1:]
    no_shop = "--no-shop" in args
    nums = [a for a in args if a != "--no-shop"]
    num_episodes = int(nums[0]) if nums else 20
    env_factory = partial(BalatroEnv, enable_shop=not no_shop)

    random_results = [run_episode(random_action, seed=i, env_factory=env_factory) for i in range(num_episodes)]
    heuristic_results = [run_episode(heuristic_action, seed=i, env_factory=env_factory) for i in range(num_episodes)]

    summarize("random", random_results)
    summarize("heuristic", heuristic_results)


if __name__ == "__main__":
    main()
