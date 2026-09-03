"""Targeted, narrow behavior-cloning correction on top of an existing
checkpoint: specifically teaches "play a made hand (full house/flush) you
already have, don't discard into it" -- a pattern the model gets almost no
natural exposure to during normal RL training, since a random 8-card deal
rarely already contains one (confirmed via scripts nearby: ~90-100% of the
time the model broke up a made hand it already had).

Deliberately narrow and low-magnitude to avoid the earlier BC-on-checkpoint
regression documented in train_warmstart.py's --init-checkpoint help text
(a fresh full-scale BC pass at lr=1e-3 sent eval reward from +2.2 to -12.4):
a handful of epochs, a learning rate ~100x smaller, and a dataset that mixes
in generic heuristic-labeled examples alongside the made-hand-specific ones
so the correction doesn't overfit to one narrow pattern.

The "correct" action for every example is computed by the project's own
exhaustive-search heuristic (agents.heuristic_action / _best_combo), not
hardcoded -- for a made-hand scenario this will naturally select the made
hand itself, since it's the highest-scoring available play.

Usage: python scripts/bc_correction_made_hand.py [--checkpoint PATH]
    [--out PATH] [--n-made-hand N] [--n-generic N] [--epochs N] [--lr LR]
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from stable_baselines3 import PPO

from balatro_sim.agents import _best_combo, heuristic_action
from balatro_sim.env import HAND_SLOTS, BalatroEnv
from balatro_sim.game_state import GameState
from balatro_sim.hands import HandType


def made_hand_example(rng: random.Random, hand_type: HandType) -> tuple[np.ndarray, np.ndarray] | None:
    game = GameState(rng=random.Random(rng.randint(0, 2**31 - 1)))
    game.force_made_hand_for_training(hand_type)
    game.hands_remaining = rng.choice([1, 2, 3, 4])
    game.discards_remaining = rng.choice([0, 1, 2, 3, 4])

    env = BalatroEnv()
    env.game = game
    obs = env._get_obs()

    best_cards, _ = _best_combo(game)
    if not best_cards:
        return None
    indices = [game.hand.index(c) for c in best_cards]
    action = np.zeros(HAND_SLOTS + 2, dtype=np.int64)
    for i in indices:
        action[i] = 1
    action[HAND_SLOTS] = 0  # mode=0 -> play
    return obs, action


def generic_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    env = BalatroEnv()
    obs, _ = env.reset(seed=rng.randint(0, 2**31 - 1))
    action = heuristic_action(env, obs)
    return obs, np.asarray(action, dtype=np.int64)


def build_dataset(n_made_hand: int, n_generic: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    hand_types = [HandType.FULL_HOUSE, HandType.FLUSH]
    made, attempts = 0, 0
    while made < n_made_hand and attempts < n_made_hand * 3:
        attempts += 1
        result = made_hand_example(rng, hand_types[made % 2])
        if result is None:
            continue
        obs, action = result
        obs_list.append(obs)
        action_list.append(action)
        made += 1

    for _ in range(n_generic):
        obs, action = generic_example(rng)
        obs_list.append(obs)
        action_list.append(action)

    return np.array(obs_list, dtype=np.float32), np.array(action_list, dtype=np.int64)


def behavior_clone(model: PPO, obs: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int, lr: float) -> None:
    device = model.policy.device
    obs_t = torch.as_tensor(obs, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    n = obs_t.shape[0]
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            _, log_prob, _ = model.policy.evaluate_actions(obs_t[idx], actions_t[idx])
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"[bc-correction] epoch {epoch + 1}/{epochs} loss={total_loss / n_batches:.4f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/exp9_full_ante1.zip")
    parser.add_argument("--out", default="checkpoints/exp12_made_hand_bc.zip")
    parser.add_argument("--n-made-hand", type=int, default=2000)
    parser.add_argument("--n-generic", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"loading {args.checkpoint}", flush=True)
    model = PPO.load(args.checkpoint, device="cpu")

    print(f"building dataset: {args.n_made_hand} made-hand + {args.n_generic} generic examples", flush=True)
    obs, actions = build_dataset(args.n_made_hand, args.n_generic, args.seed)
    print(f"collected {len(obs)} examples", flush=True)

    behavior_clone(model, obs, actions, args.epochs, args.batch_size, args.lr)
    model.save(args.out)
    print(f"saved corrected checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
