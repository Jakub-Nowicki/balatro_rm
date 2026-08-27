from __future__ import annotations

import random
from dataclasses import dataclass

from balatro_sim.blinds import BLIND_ORDER, Blind, blind_requirement, blind_reward as get_blind_reward
from balatro_sim.cards import Card, Deck
from balatro_sim.economy import UNUSED_HAND_BONUS, calculate_interest
from balatro_sim.hands import evaluate_hand
from balatro_sim.jokers import Joker
from balatro_sim.scoring import ScoreResult, score_hand

@dataclass
class CashOutResult:
    blind_reward: int
    hand_bonus: int
    interest: int
    total: int


STARTING_MONEY = 4
HAND_SIZE = 8
HANDS_PER_ROUND = 4
DISCARDS_PER_ROUND = 3  # Red Deck (+1 discard) is applied by callers, not hardcoded here


class GameState:
    def __init__(
        self,
        ante: int = 1,
        blind: Blind = Blind.SMALL,
        hand_size: int = HAND_SIZE,
        hands_per_round: int = HANDS_PER_ROUND,
        discards_per_round: int = DISCARDS_PER_ROUND,
        starting_money: int = STARTING_MONEY,
        jokers: list[Joker] | None = None,
        rng: random.Random | None = None,
    ):
        self.ante = ante
        self.blind = blind
        self.hand_size = hand_size
        self.hands_per_round = hands_per_round
        self.discards_per_round = discards_per_round
        self.money = starting_money
        self.jokers = jokers if jokers is not None else []
        self.rng = rng or random.Random()
        self._start_round()

    def _start_round(self) -> None:
        self.deck = Deck()
        self.deck.shuffle(self.rng)
        self.hand: list[Card] = []
        self.hands_remaining = self.hands_per_round
        self.discards_remaining = self.discards_per_round
        self.round_chips = 0
        self._draw_up_to_hand_size()

    def _draw_up_to_hand_size(self) -> None:
        need = self.hand_size - len(self.hand)
        if need > 0:
            self.hand.extend(self.deck.draw(need))

    @property
    def requirement(self) -> int:
        return blind_requirement(self.ante, self.blind)

    @property
    def is_blind_beaten(self) -> bool:
        return self.round_chips >= self.requirement

    @property
    def is_round_over(self) -> bool:
        return self.is_blind_beaten or self.hands_remaining == 0

    @property
    def is_game_over_loss(self) -> bool:
        return self.hands_remaining == 0 and not self.is_blind_beaten

    def play(self, cards: list[Card]) -> ScoreResult:
        if self.hands_remaining <= 0:
            raise ValueError("no hands remaining this round")
        if not cards or len(cards) > 5:
            raise ValueError("must play between 1 and 5 cards")
        for c in cards:
            if c not in self.hand:
                raise ValueError(f"{c!r} is not in hand")

        result = evaluate_hand(cards)
        score = score_hand(
            result,
            played_cards=cards,
            jokers=self.jokers,
            hands_remaining=self.hands_remaining,
            discards_remaining=self.discards_remaining,
            money=self.money,
        )
        self.round_chips += score.total

        for c in cards:
            self.hand.remove(c)
        self.hands_remaining -= 1
        self._draw_up_to_hand_size()
        return score

    def discard(self, cards: list[Card]) -> None:
        if self.discards_remaining <= 0:
            raise ValueError("no discards remaining this round")
        if not cards or len(cards) > 5:
            raise ValueError("must discard between 1 and 5 cards")
        for c in cards:
            if c not in self.hand:
                raise ValueError(f"{c!r} is not in hand")

        for c in cards:
            self.hand.remove(c)
        self.discards_remaining -= 1
        self._draw_up_to_hand_size()

    def collect_reward_and_advance(self) -> CashOutResult:
        if not self.is_blind_beaten:
            raise ValueError("cannot advance: blind not beaten")

        blind_reward = get_blind_reward(self.blind)
        hand_bonus = self.hands_remaining * UNUSED_HAND_BONUS
        interest = calculate_interest(self.money)
        total = blind_reward + hand_bonus + interest
        self.money += total

        idx = BLIND_ORDER.index(self.blind)
        if idx + 1 < len(BLIND_ORDER):
            self.blind = BLIND_ORDER[idx + 1]
        else:
            self.blind = Blind.SMALL
            self.ante += 1

        self._start_round()
        return CashOutResult(blind_reward=blind_reward, hand_bonus=hand_bonus, interest=interest, total=total)
