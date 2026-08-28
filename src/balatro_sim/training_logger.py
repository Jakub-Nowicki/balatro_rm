from __future__ import annotations

import csv
import os
import time

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeCsvLogger(BaseCallback):
    """Appends one row per finished training episode to a CSV file.

    Requires the env to be wrapped in stable_baselines3.common.monitor.Monitor,
    which populates info["episode"] = {"r": total_reward, "l": length} when an
    episode ends. Opens in Excel/Sheets directly -- no extra dependency needed.
    """

    FIELDS = ["episode", "timestep", "reward", "length", "ante", "result"]

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode_count = 0

    def _on_training_start(self) -> None:
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.episode_count += 1
            row = [
                self.episode_count,
                self.num_timesteps,
                episode["r"],
                episode["l"],
                info.get("ante", ""),
                info.get("result", ""),
            ]
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        return True


class ProgressPrinter(BaseCallback):
    """Prints one plain status line every `print_every_pct`% of training.

    Deliberately not tqdm's carriage-return progress bar: when stdout is
    piped to a log file (e.g. a backgrounded training run) tqdm's \\r updates
    turn into a wall of near-duplicate lines instead of one moving bar. This
    just prints a fresh line periodically, which reads fine either way.
    """

    def __init__(self, total_timesteps: int, print_every_pct: float = 5.0, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_every = max(1, int(total_timesteps * print_every_pct / 100))
        self._next_print = self.print_every
        self._start_time = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Vectorized envs advance num_timesteps by n_envs per call, so it can
        # jump straight past _next_print (or, once training overshoots
        # total_timesteps to finish a rollout, stay past it forever). Setting
        # the next threshold relative to *now* rather than incrementing the
        # old one avoids firing on every remaining call once that happens.
        if self.num_timesteps >= self._next_print:
            elapsed = time.time() - self._start_time
            pct = 100 * self.num_timesteps / self.total_timesteps
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            eta = max(0.0, (self.total_timesteps - self.num_timesteps) / fps) if fps > 0 else 0.0
            print(
                f"[progress] {self.num_timesteps}/{self.total_timesteps} ({pct:.1f}%) "
                f"fps={fps:.0f} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
            self._next_print = self.num_timesteps + self.print_every
        return True
