import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from balatro_sim.env import BalatroEnv
from balatro_sim.training_logger import EpisodeCsvLogger, ProgressPrinter


def test_episode_csv_logger_writes_header_and_rows(tmp_path):
    csv_path = str(tmp_path / "training_log.csv")
    vec_env = DummyVecEnv([lambda: Monitor(BalatroEnv())])
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=64, batch_size=32)

    model.learn(total_timesteps=512, callback=EpisodeCsvLogger(csv_path))

    assert os.path.exists(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == EpisodeCsvLogger.FIELDS
    assert len(rows) > 1  # at least one episode should have finished in 512 steps

    episode_row = rows[1]
    episode_num, timestep, reward, length, ante, result = episode_row
    assert int(episode_num) == 1
    assert int(timestep) > 0
    float(reward)  # parses without error
    assert int(length) > 0
    assert result in ("lost", "won_run")


def test_progress_printer_prints_periodic_status_lines(capsys):
    vec_env = DummyVecEnv([lambda: Monitor(BalatroEnv())])
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=64, batch_size=32)

    model.learn(total_timesteps=512, callback=ProgressPrinter(total_timesteps=512, print_every_pct=25.0))

    out = capsys.readouterr().out
    progress_lines = [line for line in out.splitlines() if line.startswith("[progress]")]
    assert len(progress_lines) >= 1
    assert "fps=" in progress_lines[-1]
    assert "eta=" in progress_lines[-1]
