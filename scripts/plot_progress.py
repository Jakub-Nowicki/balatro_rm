"""Plot PPO training progress from an EpisodeCsvLogger CSV.

Usage: python scripts/plot_progress.py <csv_path> [output.png]
"""
from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int) -> list[float]:
    if len(values) < window:
        return values
    out = [float("nan")] * (window - 1)
    running_sum = sum(values[:window])
    out.append(running_sum / window)
    for i in range(window, len(values)):
        running_sum += values[i] - values[i - window]
        out.append(running_sum / window)
    return out


def load_csv(path: str) -> tuple[list[int], list[float], list[int]]:
    episodes: list[int] = []
    rewards: list[float] = []
    antes: list[int] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            antes.append(int(row["ante"]) if row["ante"] else 0)
    return episodes, rewards, antes


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_progress.py <csv_path> [output.png]")
        sys.exit(1)
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.rsplit(".", 1)[0] + ".png"

    episodes, rewards, antes = load_csv(csv_path)
    if not episodes:
        print(f"{csv_path} has no completed episodes yet")
        sys.exit(1)

    window = max(1, min(200, len(episodes) // 20 or 1))
    smoothed_reward = moving_average(rewards, window)
    smoothed_ante = moving_average([float(a) for a in antes], window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(episodes, rewards, alpha=0.15, color="tab:blue")
    ax1.plot(episodes, smoothed_reward, color="tab:blue", linewidth=2, label=f"{window}-episode moving avg")
    ax1.set_ylabel("episode reward")
    ax1.set_title(f"PPO training progress -- {csv_path}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(episodes, antes, alpha=0.15, color="tab:orange")
    ax2.plot(episodes, smoothed_ante, color="tab:orange", linewidth=2, label=f"{window}-episode moving avg")
    ax2.set_ylabel("ante reached")
    ax2.set_xlabel("episode")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"saved chart to {output_path} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
