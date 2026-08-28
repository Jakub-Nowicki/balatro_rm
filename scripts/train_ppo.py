"""Train a PPO agent on BalatroEnv, logging every episode to CSV, then
compare the trained agent against the baselines (all evaluated on the same
env config the agent trained on, for a fair comparison).

Usage: python scripts/train_ppo.py [--timesteps N] [--n-envs N] [--device auto|cpu|cuda]
                                    [--csv-path PATH] [--no-shop] [--ent-coef F]

--n-envs > 1 runs environments in parallel worker processes (SubprocVecEnv).
Since BalatroEnv's bottleneck is the pure-Python game loop, not the neural
net, this scales training throughput far more than --device cuda does for a
small MLP policy -- SB3's own device="auto" defaults MLP policies to CPU for
exactly that reason.

--no-shop collapses the env to round-only play (no joker purchases), useful
for isolating the round-phase reward signal from shop-decision complexity.

--ent-coef defaults to 0.01, not SB3's own default of 0.0. With ent_coef=0.0
a MultiDiscrete policy over this large a combinatorial action space can
converge prematurely onto a narrow, low-quality action distribution before
it has explored enough to find better ones; a small entropy bonus keeps
exploration alive longer.
"""
from __future__ import annotations

import argparse
import statistics
from functools import partial

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from balatro_sim.agents import heuristic_action, random_action, run_episode
from balatro_sim.env import BalatroEnv
from balatro_sim.training_logger import EpisodeCsvLogger, ProgressPrinter

CHECKPOINT_PATH = "checkpoints/ppo_balatro"
DEFAULT_LOG_PATH = "logs/ppo_training_log.csv"
EVAL_EPISODES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--csv-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--no-shop", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", default="64,64", help="comma-separated hidden layer sizes, e.g. 256,256")
    parser.add_argument(
        "--vec-env",
        default="subprocess",
        choices=["subprocess", "dummy"],
        help="'dummy' steps all envs sequentially in one process (no IPC) -- can beat 'subprocess' on Windows "
        "for a very cheap env, where per-step IPC round-trip overhead dominates the actual game logic cost.",
    )
    return parser.parse_args()


def _make_env(enable_shop: bool):
    # Must be a module-level function, not a closure: SubprocVecEnv pickles this
    # to send it to worker processes, and Windows' "spawn" start method can't
    # pickle a function defined inside another function.
    return Monitor(BalatroEnv(enable_shop=enable_shop))


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
    env_factory = partial(BalatroEnv, enable_shop=not args.no_shop)
    env_fns = [partial(_make_env, enable_shop=not args.no_shop) for _ in range(max(1, args.n_envs))]

    if args.vec_env == "dummy":
        vec_env = DummyVecEnv(env_fns)
    elif args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
        if args.n_envs >= 16:
            # PyTorch multi-threads its own CPU ops by default. With N
            # already-parallel worker processes, the main process doing that
            # too can oversubscribe the CPU -- confirmed at n_envs=24 (fps
            # collapsed ~1400 -> ~290). But NOT at n_envs=8, where 24 idle
            # threads are still available for torch to use productively;
            # forcing 1 thread there measured ~3x *slower* (880 -> ~300fps).
            # Only restrict once there isn't much headroom left for it.
            torch.set_num_threads(1)
    else:
        vec_env = DummyVecEnv(env_fns)

    net_arch = [int(x) for x in args.net_arch.split(",")]
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=args.device,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=net_arch),
    )
    callback = CallbackList([EpisodeCsvLogger(args.csv_path), ProgressPrinter(args.timesteps)])
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
    summarize("ppo", ppo_results)


if __name__ == "__main__":
    main()
