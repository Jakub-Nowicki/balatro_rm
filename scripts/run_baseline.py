"""Run random vs. heuristic agents through BalatroEnv and print summary stats.

Usage: python scripts/run_baseline.py [num_episodes]
"""
from __future__ import annotations

import statistics
import sys

from balatro_sim.agents import heuristic_action, random_action, run_episode


def summarize(name: str, results: list[dict]) -> None:
    rewards = [r["reward"] for r in results]
    antes = [r["ante"] for r in results]
    wins = sum(1 for r in results if r["result"] == "won_run")
    print(f"{name}: mean_reward={statistics.mean(rewards):.3f} mean_ante={statistics.mean(antes):.2f} wins={wins}/{len(results)}")


def main() -> None:
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    random_results = [run_episode(random_action, seed=i) for i in range(num_episodes)]
    heuristic_results = [run_episode(heuristic_action, seed=i) for i in range(num_episodes)]

    summarize("random", random_results)
    summarize("heuristic", heuristic_results)


if __name__ == "__main__":
    main()
