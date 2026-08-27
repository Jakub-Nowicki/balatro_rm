from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_sim.blinds import BLIND_ORDER
from balatro_sim.cards import Card, Suit
from balatro_sim.game_state import GameState
from balatro_sim.jokers import Joker

HAND_SLOTS = 8
MAX_ANTE = 8  # ANTE_BASE_CHIPS only covers 1-8; reaching ante 9 ends the episode as a win
CARD_FEATURES = 5  # 1 normalized rank + 4 one-hot suit
OBS_SIZE = HAND_SLOTS * CARD_FEATURES + 7

INVALID_ACTION_PENALTY = -0.05
BLIND_BEATEN_BONUS = 0.5
RUN_WON_BONUS = 5.0
LOSS_PENALTY = -1.0


class BalatroEnv(gym.Env):
    """A minimal Gym-style wrapper around GameState.

    Action: MultiDiscrete([2]*8 + [2]) -- 8 binary flags selecting hand slots,
    plus a trailing mode flag (0 = play, 1 = discard). Shop/joker purchases are
    not exposed yet; jokers are fixed for the whole episode via the constructor.

    Observation: a flat float vector -- 8 hand slots x (rank, one-hot suit),
    followed by hands_remaining, discards_remaining, round-progress fraction,
    normalized money, normalized ante, normalized blind index, joker count.
    """

    metadata = {"render_modes": []}

    def __init__(self, jokers: list[Joker] | None = None):
        super().__init__()
        self._starting_jokers = jokers or []
        self.action_space = spaces.MultiDiscrete([2] * HAND_SLOTS + [2])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.game: GameState | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = random.Random(seed) if seed is not None else random.Random()
        self.game = GameState(jokers=list(self._starting_jokers), rng=rng)
        return self._get_obs(), {}

    @staticmethod
    def _encode_card(card: Card) -> list[float]:
        rank_norm = (card.rank.value - 2) / 12.0
        suit_onehot = [1.0 if card.suit is s else 0.0 for s in Suit]
        return [rank_norm, *suit_onehot]

    def _get_obs(self) -> np.ndarray:
        game = self.game
        feats: list[float] = []
        for i in range(HAND_SLOTS):
            if i < len(game.hand):
                feats.extend(self._encode_card(game.hand[i]))
            else:
                feats.extend([0.0] * CARD_FEATURES)

        progress = min(1.0, game.round_chips / game.requirement) if game.ante <= MAX_ANTE else 1.0
        feats.append(game.hands_remaining / game.hands_per_round)
        feats.append(game.discards_remaining / game.discards_per_round)
        feats.append(progress)
        feats.append(min(1.0, game.money / 50.0))
        feats.append(min(1.0, game.ante / MAX_ANTE))
        feats.append(BLIND_ORDER.index(game.blind) / (len(BLIND_ORDER) - 1))
        feats.append(min(1.0, len(game.jokers) / 5.0))
        return np.array(feats, dtype=np.float32)

    def step(self, action):
        game = self.game
        selection, mode = action[:HAND_SLOTS], action[HAND_SLOTS]
        indices = [i for i, flag in enumerate(selection) if flag and i < len(game.hand)]
        cards = [game.hand[i] for i in indices]

        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        valid = 1 <= len(cards) <= 5
        if mode == 0:
            valid = valid and game.hands_remaining > 0
        else:
            valid = valid and game.discards_remaining > 0

        if not valid:
            reward = INVALID_ACTION_PENALTY
        elif mode == 0:
            score = game.play(cards)
            reward = score.total / max(1, game.requirement)
            if game.is_blind_beaten:
                game.collect_reward_and_advance()
                reward += BLIND_BEATEN_BONUS
                if game.ante > MAX_ANTE:
                    reward += RUN_WON_BONUS
                    terminated, truncated = True, True
                    info["result"] = "won_run"
            elif game.is_game_over_loss:
                reward += LOSS_PENALTY
                terminated = True
                info["result"] = "lost"
        else:
            game.discard(cards)

        return self._get_obs(), reward, terminated, truncated, info
