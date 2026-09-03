from __future__ import annotations

import random
from itertools import combinations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_sim.blinds import BLIND_ORDER, BossBlind
from balatro_sim.cards import Card, Suit
from balatro_sim.game_state import DISCARDS_PER_ROUND, GameState
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.jokers import JOKER_NAME_TO_INDEX, JOKER_NAMES, Joker
from balatro_sim.scoring import score_hand
from balatro_sim.shop import CARD_SLOTS as SHOP_SLOTS

HAND_SLOTS = 8
MAX_ANTE = 8  # ante base chips table only covers 1-8
CARD_FEATURES = 5  # 1 normalized rank + 4 one-hot suit
SHOP_ACTION_SIZE = SHOP_SLOTS + 2  # buy offering[i], ..., reroll, leave shop
MAX_JOKER_PRICE_NORM = 20.0
MAX_REROLL_COST_NORM = 20.0
MAX_ACHIEVABLE_HAND_TYPE = HandType.STRAIGHT_FLUSH.value  # highest hand type reachable with no card enhancements
OBS_SIZE = HAND_SLOTS * CARD_FEATURES + 7 + 1 + HAND_SLOTS + 1 + SHOP_SLOTS * 2 + 1

INVALID_ACTION_PENALTY = -0.05
BLIND_BEATEN_BONUS = 0.3
RUN_WON_BONUS = 2.0
LOSS_PENALTY = -1.0
MAX_CONSECUTIVE_INVALID = 5
DISCARD_COST = -0.02  # small cost so discards aren't treated as free
# Reward bonus for playing close to the best possible hand from the current cards
# (1.0 = played optimally). This is what teaches the agent to always play its best hand.
HAND_EFFICIENCY_WEIGHT = 1.0


def _best_achievable_score_total(game: GameState, hand: list[Card]) -> int:
    """Highest score achievable from any legal subset of hand, under the current
    scoring context (jokers, boss, deck). Used for the hand-efficiency reward."""
    debuffed_suit = game.active_boss.debuffed_suit if game.active_boss else None
    debuffed_cards = (
        game.cards_played_this_ante
        if game.active_boss and game.active_boss.debuffs_previously_played_cards
        else None
    )
    required_size = game.active_boss.required_play_size if game.active_boss else None
    sizes = [required_size] if required_size is not None else range(1, min(5, len(hand)) + 1)
    best = 0
    for k in sizes:
        if k > len(hand):
            continue
        for combo in combinations(hand, k):
            result = evaluate_hand(list(combo))
            score = score_hand(
                result,
                played_cards=list(combo),
                jokers=game.jokers,
                hands_remaining=game.hands_remaining,
                discards_remaining=game.discards_remaining,
                money=game.money,
                debuffed_suit=debuffed_suit,
                debuffed_cards=debuffed_cards,
                deck_size=len(game.deck),
                max_joker_slots=game.max_joker_slots,
                rng=game.rng,
            )
            if score.total > best:
                best = score.total
    return best


def _best_achievable_hand_type_and_combo(hand: list[Card]) -> tuple[HandType, list[Card]]:
    """Strongest HandType formable from any subset of hand, and the cards that
    form it. Used as two observation features: the hand type itself, and a
    per-card flag for whether that card is part of the best hand."""
    best_type = HandType.HIGH_CARD
    best_combo: list[Card] = list(hand[:1])
    for k in range(1, min(5, len(hand)) + 1):
        for combo in combinations(hand, k):
            hand_type = evaluate_hand(list(combo)).hand_type
            if hand_type.value > best_type.value:
                best_type = hand_type
                best_combo = list(combo)
    return best_type, best_combo


def _best_achievable_hand_type(hand: list[Card]) -> HandType:
    return _best_achievable_hand_type_and_combo(hand)[0]


class BalatroEnv(gym.Env):
    """A Gym-style wrapper around GameState, alternating between a round
    phase (play/discard hands) and a shop phase (buy/reroll/skip jokers)
    after every blind.

    Action: MultiDiscrete([2]*HAND_SLOTS + [2, SHOP_ACTION_SIZE]).
      - dims 0..HAND_SLOTS-1: one binary flag per hand slot, 1 = include this
        card in the play/discard. Only read during the round phase.
      - dim HAND_SLOTS: mode, 0 = play, 1 = discard.
      - dim HAND_SLOTS+1: shop choice, 0/1 = buy that offering, SHOP_SLOTS =
        reroll, SHOP_SLOTS+1 = leave the shop. Only read during the shop phase.

    Observation: a flat float vector. 8 hand slots x (rank, one-hot suit),
    hands/discards remaining, round progress, normalized money/ante/blind
    index, joker-slot fill fraction, the best achievable hand type, 8 flags
    for which cards form that hand, a shop-phase flag, up to SHOP_SLOTS shop
    offerings (price, joker identity), and normalized reroll cost.

    enable_shop=False skips the shop phase entirely (auto-leaves with no
    reward), useful for training on round-play only.

    win_at_ante=1 (default) ends the episode as a win once Ante 1's Boss
    Blind is cleared. Multi-ante boss pools aren't implemented yet, so this
    shouldn't be raised above 1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jokers: list[Joker] | None = None,
        enable_shop: bool = True,
        win_at_ante: int = 1,
        boss_pool: list[BossBlind] | None = None,
        made_hand_bias: dict[str, float] | None = None,
    ):
        super().__init__()
        self._starting_jokers = jokers or []
        self.enable_shop = enable_shop
        self.win_at_ante = win_at_ante
        self.boss_pool = boss_pool
        # Training-only: e.g. {"FULL_HOUSE": 0.1} deals that hand type already
        # complete some fraction of rounds, since it rarely happens naturally.
        self.made_hand_bias = made_hand_bias
        self.action_space = spaces.MultiDiscrete([2] * HAND_SLOTS + [2, SHOP_ACTION_SIZE])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.game: GameState | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = random.Random(seed) if seed is not None else random.Random()
        self.game = GameState(
            jokers=list(self._starting_jokers),
            discards_per_round=DISCARDS_PER_ROUND + 1,  # Red Deck gives +1 discard
            boss_pool=self.boss_pool,
            rng=rng,
        )
        self._consecutive_invalid = 0
        self._maybe_bias_hand()
        return self._get_obs(), {}

    def _maybe_bias_hand(self) -> None:
        """Training-only: with the configured probability, forces this
        round's starting hand to already contain a made FULL_HOUSE or FLUSH.
        No-op when made_hand_bias isn't set (the default)."""
        if not self.made_hand_bias:
            return
        roll = self.game.rng.random()
        cumulative = 0.0
        for type_name, prob in self.made_hand_bias.items():
            cumulative += prob
            if roll < cumulative:
                self.game.force_made_hand_for_training(HandType[type_name])
                return

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
        feats.append(min(1.0, len(game.jokers) / game.max_joker_slots))

        if game.phase == "round" and game.hand:
            hand = game.hand
            best_type, best_combo = _best_achievable_hand_type_and_combo(hand)
            feats.append(min(1.0, best_type.value / MAX_ACHIEVABLE_HAND_TYPE))
            best_combo_set = set(best_combo)
            for i in range(HAND_SLOTS):
                feats.append(1.0 if i < len(hand) and hand[i] in best_combo_set else 0.0)
        else:
            feats.append(0.0)
            feats.extend([0.0] * HAND_SLOTS)

        in_shop = game.phase == "shop" and game.shop is not None
        feats.append(1.0 if in_shop else 0.0)
        offerings = game.shop.offerings if in_shop else []
        for i in range(SHOP_SLOTS):
            if i < len(offerings):
                item = offerings[i]
                feats.append(min(1.0, item.price / MAX_JOKER_PRICE_NORM))
                feats.append(JOKER_NAME_TO_INDEX.get(item.name, 0) / max(1, len(JOKER_NAMES) - 1))
            else:
                feats.extend([0.0, 0.0])
        reroll_cost = game.shop.reroll_cost if in_shop else 0
        feats.append(min(1.0, reroll_cost / MAX_REROLL_COST_NORM))

        return np.array(feats, dtype=np.float32)

    def step(self, action):
        game = self.game
        blind_before = (game.ante, game.blind)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        if game.phase == "round":
            mode = int(action[HAND_SLOTS])
            # Drop flags pointing past the current hand size (can happen under a
            # boss that shrinks hand size).
            selected = [i for i in range(HAND_SLOTS) if action[i] and i < len(game.hand)]
            if not selected:
                # Nothing valid was flagged, so fall back to the single highest card
                # instead of just rejecting the action.
                selected = [max(range(len(game.hand)), key=lambda i: game.hand[i].rank.chip_value)]
            elif len(selected) > 5:
                selected = selected[:5]
            cards = [game.hand[i] for i in selected]

            # Discarding with 0 discards left falls back to playing instead,
            # so the round always keeps moving forward.
            if mode == 1 and game.discards_remaining <= 0:
                mode = 0

            valid = True
            if mode == 0:
                valid = game.hands_remaining > 0
                # The Psychic boss requires playing exactly 5 cards.
                required = game.active_boss.required_play_size if game.active_boss else None
                if required is not None:
                    valid = valid and len(cards) == required
            else:
                valid = game.discards_remaining > 0

            if not valid:
                reward = INVALID_ACTION_PENALTY
            elif mode == 0:
                # Computed before play() so it uses the same context (hands
                # left, discards left, etc.) the real play was scored under.
                best_total = _best_achievable_score_total(game, list(game.hand))
                score = game.play(cards)
                reward = score.total / max(1, game.requirement)
                if best_total > 0:
                    reward += HAND_EFFICIENCY_WEIGHT * (score.total / best_total)
                if game.is_blind_beaten:
                    game.collect_reward_and_advance()  # enters "shop" phase
                    reward += BLIND_BEATEN_BONUS
                    if game.ante > self.win_at_ante:
                        reward += RUN_WON_BONUS
                        terminated, truncated = True, True
                        info["result"] = "won_run"
                    elif not self.enable_shop:
                        game.leave_shop()  # skip within the same step
                elif game.is_game_over_loss:
                    reward += LOSS_PENALTY
                    terminated = True
                    info["result"] = "lost"
            else:
                game.discard(cards)
                reward = DISCARD_COST
        elif not self.enable_shop:
            game.leave_shop()
        else:  # phase == "shop"
            shop_choice = int(action[HAND_SLOTS + 1])
            try:
                offerings = game.shop.offerings
                if shop_choice < SHOP_SLOTS:
                    if shop_choice >= len(offerings):
                        raise ValueError("no offering at that slot")
                    game.buy_joker(offerings[shop_choice])
                elif shop_choice == SHOP_SLOTS:
                    game.reroll_shop()
                else:
                    game.leave_shop()
            except ValueError:
                reward = INVALID_ACTION_PENALTY

        # Force an exit if the agent keeps repeating an invalid shop action,
        # so a badly-trained shop policy can't get stuck forever.
        if reward == INVALID_ACTION_PENALTY:
            self._consecutive_invalid += 1
        else:
            self._consecutive_invalid = 0
        if self._consecutive_invalid >= MAX_CONSECUTIVE_INVALID and game.phase == "shop":
            game.leave_shop()
            self._consecutive_invalid = 0

        if game.phase == "round" and (game.ante, game.blind) != blind_before:
            self._maybe_bias_hand()

        if terminated or truncated:
            info["ante"] = game.ante

        return self._get_obs(), reward, terminated, truncated, info
