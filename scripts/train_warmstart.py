"""Behavior-clones the heuristic agent into a PPO policy, then continues
training with real PPO on the actual reward signal (a "warm start").

Three phases, all logged so progress can be checked mid run:
  1. collect_bc_dataset: runs the heuristic and records every (obs, action)
     pair it produces. Prints "[bc-collect] i/N episodes" periodically.
  2. behavior_clone: supervised training of the policy network to imitate
     those pairs. Prints "[bc-train] epoch i/N loss=..."
  3. model.learn: normal PPO fine-tuning from the warm-started weights.
     Prints "[progress] step/total (%)" periodically and logs every episode
     to --csv-path.

Usage: python scripts/train_warmstart.py [--bc-episodes N] [--bc-epochs N]
    [--timesteps N] [--n-envs N] [--device auto|cpu|cuda] [--no-shop]
    [--net-arch 64,64] [--ent-coef 0.01] [--learning-rate 3e-4] [--lr-schedule constant|linear]
    [--checkpoint-every N] [--csv-path PATH] [--checkpoint-path PATH]
"""
from __future__ import annotations

import argparse
import statistics
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from balatro_sim.agents import heuristic_action, random_action, run_episode
from balatro_sim.blinds import IMPLEMENTED_ANTE_1_BOSSES
from balatro_sim.env import BalatroEnv
from balatro_sim.training_logger import EpisodeCsvLogger, ProgressPrinter

CHECKPOINT_PATH = "checkpoints/ppo_warmstart"
DEFAULT_LOG_PATH = "logs/ppo_warmstart_log.csv"
EVAL_EPISODES = 30
MAX_STEPS_PER_EPISODE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="load an existing .zip checkpoint's weights instead of building a fresh network. "
        "Useful when the loaded model already learned a related skill and just needs to learn "
        "what's new. Requires matching observation and action space shapes.",
    )
    parser.add_argument("--bc-episodes", type=int, default=5000)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=15_000_000)
    parser.add_argument("--n-envs", type=int, default=24)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--csv-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--no-shop", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", default="64,64")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        default="constant",
        choices=["constant", "linear"],
        help="'linear' decays --learning-rate down to 0 over the course of training, so updates "
        "get smaller and more precise later on instead of staying just as aggressive throughout.",
    )
    parser.add_argument(
        "--vec-env",
        default="subprocess",
        choices=["subprocess", "dummy"],
        help="'dummy' steps all envs sequentially in one process instead of using separate worker "
        "processes. Can be faster on Windows when each step is cheap.",
    )
    parser.add_argument(
        "--bias-bosses",
        default=None,
        help="comma-separated Boss Blind names (e.g. 'The Psychic,The Pillar') to draw more often "
        "than the rest, for practicing against bosses a policy is weak against.",
    )
    parser.add_argument(
        "--bias-multiplier",
        type=int,
        default=3,
        help="how many extra copies of each --bias-bosses entry to add to the boss pool.",
    )
    parser.add_argument(
        "--made-hand-bias",
        default=None,
        help="comma-separated hand types (FULL_HOUSE, FLUSH) to deal already-complete in the "
        "starting hand more often than they occur naturally, since a random deal rarely already "
        "contains one and the policy needs practice recognizing them.",
    )
    parser.add_argument(
        "--made-hand-bias-prob",
        type=float,
        default=0.1,
        help="probability per round, for EACH --made-hand-bias entry, of forcing that hand type.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200_000,
        help="save an intermediate checkpoint every N timesteps, in addition to the final save, so "
        "an unexpected shutdown mid run doesn't lose all progress.",
    )
    return parser.parse_args()


def linear_schedule(initial_value: float):
    """SB3 passes progress_remaining=1.0 at the start of training, 0.0 at the end."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def _make_env(enable_shop: bool, boss_pool=None, made_hand_bias=None):
    # Module-level, not a closure: SubprocVecEnv pickles this for Windows' spawn.
    return Monitor(BalatroEnv(enable_shop=enable_shop, boss_pool=boss_pool, made_hand_bias=made_hand_bias))


def _build_boss_pool(bias_bosses: str | None, multiplier: int):
    if not bias_bosses:
        return None
    names = {n.strip() for n in bias_bosses.split(",")}
    unknown = names - {b.name for b in IMPLEMENTED_ANTE_1_BOSSES}
    if unknown:
        raise ValueError(f"unknown boss name(s) in --bias-bosses: {sorted(unknown)}")
    pool = list(IMPLEMENTED_ANTE_1_BOSSES)
    for boss in IMPLEMENTED_ANTE_1_BOSSES:
        if boss.name in names:
            pool.extend([boss] * multiplier)
    return pool


def _build_made_hand_bias(made_hand_bias: str | None, prob: float):
    if not made_hand_bias:
        return None
    valid = {"FULL_HOUSE", "FLUSH"}
    types = {t.strip().upper() for t in made_hand_bias.split(",")}
    unknown = types - valid
    if unknown:
        raise ValueError(f"unknown hand type(s) in --made-hand-bias: {sorted(unknown)} (valid: {sorted(valid)})")
    return {t: prob for t in types}


def collect_bc_dataset(env_factory, num_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Runs the heuristic agent and records every (obs, action) pair it produces."""
    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    report_every = max(1, num_episodes // 10)
    for ep in range(num_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=ep)
        for _ in range(MAX_STEPS_PER_EPISODE):
            action = heuristic_action(env, obs)
            obs_list.append(obs.copy())
            action_list.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if (ep + 1) % report_every == 0:
            print(f"[bc-collect] {ep + 1}/{num_episodes} episodes ({len(obs_list)} steps so far)", flush=True)
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
        print(f"[bc-train] epoch {epoch + 1}/{epochs} loss={total_loss / n_batches:.4f}", flush=True)


def ppo_action(model):
    def agent_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return agent_fn


def summarize(name: str, results: list[dict]) -> None:
    rewards = [r["reward"] for r in results]
    antes = [r["ante"] for r in results]
    wins = sum(1 for r in results if r["result"] == "won_run")
    print(f"{name}: mean_reward={statistics.mean(rewards):.3f} mean_ante={statistics.mean(antes):.2f} wins={wins}/{len(results)}")


def main() -> None:
    args = parse_args()
    boss_pool = _build_boss_pool(args.bias_bosses, args.bias_multiplier)
    if boss_pool is not None:
        print(f"biasing boss pool: {args.bias_bosses} x{args.bias_multiplier} copies each", flush=True)
    made_hand_bias = _build_made_hand_bias(args.made_hand_bias, args.made_hand_bias_prob)
    if made_hand_bias is not None:
        print(f"biasing made-hand deals: {made_hand_bias}", flush=True)
    env_factory = partial(
        BalatroEnv, enable_shop=not args.no_shop, boss_pool=boss_pool, made_hand_bias=made_hand_bias
    )

    if not args.init_checkpoint:
        print(f"=== Phase 1: collecting {args.bc_episodes} heuristic episodes for behavior cloning ===", flush=True)
        obs, actions = collect_bc_dataset(env_factory, args.bc_episodes)
        print(f"collected {len(obs)} (obs, action) pairs", flush=True)

    net_arch = [int(x) for x in args.net_arch.split(",")]
    env_fns = [
        partial(_make_env, enable_shop=not args.no_shop, boss_pool=boss_pool, made_hand_bias=made_hand_bias)
        for _ in range(max(1, args.n_envs))
    ]
    if args.vec_env == "dummy":
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    if args.vec_env != "dummy" and args.n_envs >= 16:
        # With many worker processes already using the CPU, let torch use just
        # one thread instead of oversubscribing all cores.
        torch.set_num_threads(1)

    lr = linear_schedule(args.learning_rate) if args.lr_schedule == "linear" else args.learning_rate
    if args.init_checkpoint:
        print(f"loading initial weights from {args.init_checkpoint}", flush=True)
        model = PPO.load(args.init_checkpoint, env=vec_env, device=args.device)
        model.ent_coef = args.ent_coef
        model.learning_rate = lr
        model._setup_lr_schedule()
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            device=args.device,
            ent_coef=args.ent_coef,
            learning_rate=lr,
            policy_kwargs=dict(net_arch=net_arch),
        )

    if args.init_checkpoint:
        # Skip BC when resuming from a checkpoint: BC's optimizer is tuned for
        # training from scratch and can wreck already fine-tuned weights.
        print("=== Phase 2 skipped (--init-checkpoint): going straight to RL fine-tuning ===", flush=True)
    else:
        print(f"=== Phase 2: behavior cloning the policy for {args.bc_epochs} epochs ===", flush=True)
        behavior_clone(model, obs, actions, args.bc_epochs, args.bc_batch_size, args.bc_lr)
        model.save(args.checkpoint_path + "_bc_only")

        print("=== evaluating warm-started policy before RL fine-tuning ===", flush=True)
        warmstart_results = [run_episode(ppo_action(model), seed=i, env_factory=env_factory) for i in range(EVAL_EPISODES)]
        summarize("ppo (after BC, before RL)", warmstart_results)

    print(f"=== Phase 3: PPO fine-tuning for {args.timesteps} timesteps ===", flush=True)
    model.verbose = 1
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_every // max(1, args.n_envs), 1),
        save_path="checkpoints/",
        name_prefix=args.checkpoint_path.split("/")[-1] + "_step",
    )
    callback = CallbackList([EpisodeCsvLogger(args.csv_path), ProgressPrinter(args.timesteps), checkpoint_callback])
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.checkpoint_path)
    vec_env.close()
    print(f"\nper-episode training log written to {args.csv_path}")

    ppo_results = [run_episode(ppo_action(model), seed=i, env_factory=env_factory) for i in range(EVAL_EPISODES)]
    random_results = [run_episode(random_action, seed=i, env_factory=env_factory) for i in range(EVAL_EPISODES)]
    heuristic_results = [run_episode(heuristic_action, seed=i, env_factory=env_factory) for i in range(EVAL_EPISODES)]

    print()
    summarize("random", random_results)
    summarize("heuristic", heuristic_results)
    summarize("ppo (warm start + RL)", ppo_results)


if __name__ == "__main__":
    main()
