from __future__ import annotations

from itertools import combinations

import numpy as np

from balatro_sim.env import SHOP_SLOTS, SUBSET_TO_INDEX, BalatroEnv
from balatro_sim.game_state import GameState
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.scoring import ScoreResult, score_hand

MAX_STEPS_PER_EPISODE = 500


def random_action(env: BalatroEnv, obs=None) -> np.ndarray:
    return env.action_space.sample()


def _best_combo(game: GameState) -> tuple[list, ScoreResult]:
    best_cards: list = []
    best_score = ScoreResult(chips=0, mult=0, total=-1)
    for k in range(1, min(5, len(game.hand)) + 1):
        for combo in combinations(game.hand, k):
            result = evaluate_hand(list(combo))
            score = score_hand(
                result,
                played_cards=list(combo),
                jokers=game.jokers,
                hands_remaining=game.hands_remaining,
                discards_remaining=game.discards_remaining,
                money=game.money,
            )
            if score.total > best_score.total:
                best_cards, best_score = list(combo), score
    return best_cards, best_score


def _heuristic_shop_choice(game: GameState) -> int:
    """Buys the cheapest affordable offering if a joker slot is free, else leaves."""
    LEAVE = SHOP_SLOTS + 1
    if game.shop is None or len(game.jokers) >= game.max_joker_slots:
        return LEAVE
    affordable = [(i, item) for i, item in enumerate(game.shop.offerings) if item.price <= game.money]
    if not affordable:
        return LEAVE
    cheapest_idx, _ = min(affordable, key=lambda pair: pair[1].price)
    return cheapest_idx


def heuristic_action(env: BalatroEnv, obs=None) -> np.ndarray:
    """Round phase: enumerates every 1-5 card combo in hand, scores it (jokers
    included), and plays the best one -- unless it's only High Card and a
    discard is available, in which case it discards the three weakest cards.
    Shop phase: buys the cheapest affordable joker if a slot is free, else
    leaves the shop. Never rerolls."""
    game = env.game
    action = np.zeros(3, dtype=int)

    if game.phase == "shop":
        action[2] = _heuristic_shop_choice(game)
        return action

    best_cards, best_score = _best_combo(game)
    best_hand_type = evaluate_hand(best_cards).hand_type

    if best_hand_type == HandType.HIGH_CARD and game.discards_remaining > 0:
        weakest = sorted(range(len(game.hand)), key=lambda i: game.hand[i].rank.chip_value)[:3]
        action[0] = SUBSET_TO_INDEX[frozenset(weakest)]
        action[1] = 1  # discard
    else:
        indices = [game.hand.index(card) for card in best_cards]
        action[0] = SUBSET_TO_INDEX[frozenset(indices)]
        action[1] = 0  # play

    return action


def run_episode(agent_fn, seed: int, env_factory=BalatroEnv) -> dict:
    """Runs one episode with agent_fn(env, obs) -> action. Returns a summary dict."""
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(MAX_STEPS_PER_EPISODE):
        action = agent_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return {"reward": total_reward, "ante": env.game.ante, "result": info.get("result", "truncated")}
    return {"reward": total_reward, "ante": env.game.ante, "result": "max_steps"}
