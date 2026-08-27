from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from balatro_sim.cards import Rank, Suit
from balatro_sim.hands import HandType
from balatro_sim.scoring import ScoringContext

EffectFn = Callable[[int, float, ScoringContext], "tuple[int, float]"]

FACE_RANKS = {Rank.JACK, Rank.QUEEN, Rank.KING}
# Odd Todd treats Ace as odd (it anchors the low end of A-2-3-4-5 straights).
ODD_RANK_VALUES = {3, 5, 7, 9, 14}


def _flat_mult(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips, mult + amount
    return fn


def _mult_per_scoring_suit(suit: Suit, amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if c.suit is suit)
        return chips, mult + amount * n
    return fn


def _mult_if_contains(hand_type: HandType, amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        if hand_type in ctx.contained_hand_types:
            return chips, mult + amount
        return chips, mult
    return fn


def _chips_if_contains(hand_type: HandType, amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        if hand_type in ctx.contained_hand_types:
            return chips + amount, mult
        return chips, mult
    return fn


def _mult_if_hand_size_le(n: int, amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        if len(ctx.played_cards) <= n:
            return chips, mult + amount
        return chips, mult
    return fn


def _chips_per_remaining_discard(amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips + amount * ctx.discards_remaining, mult
    return fn


def _mult_if_no_discards(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        if ctx.discards_remaining == 0:
            return chips, mult + amount
        return chips, mult
    return fn


def _mult_per_scoring_even_rank(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if c.rank.value % 2 == 0)
        return chips, mult + amount * n
    return fn


def _chips_per_scoring_odd_rank(amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if c.rank.value in ODD_RANK_VALUES)
        return chips + amount * n, mult
    return fn


def _chips_per_scoring_face(amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if c.rank in FACE_RANKS)
        return chips + amount * n, mult
    return fn


def _mult_per_joker(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips, mult + amount * ctx.joker_count
    return fn


@dataclass(frozen=True)
class JokerSpec:
    name: str
    cost: int
    rarity: str
    effect: EffectFn


# Verified against balatrowiki.org/w/Jokers. A curated subset spanning the main
# effect shapes (flat mult/chips, per-suit, hand-pattern-conditional, positional,
# resource-conditional) rather than the full 150-joker roster.
JOKER_CATALOG: dict[str, JokerSpec] = {}


def _register(name: str, cost: int, rarity: str, effect: EffectFn) -> None:
    JOKER_CATALOG[name] = JokerSpec(name, cost, rarity, effect)


_register("Joker", 2, "common", _flat_mult(4))
_register("Greedy Joker", 5, "common", _mult_per_scoring_suit(Suit.DIAMONDS, 3))
_register("Lusty Joker", 5, "common", _mult_per_scoring_suit(Suit.HEARTS, 3))
_register("Wrathful Joker", 5, "common", _mult_per_scoring_suit(Suit.SPADES, 3))
_register("Gluttonous Joker", 5, "common", _mult_per_scoring_suit(Suit.CLUBS, 3))
_register("Jolly Joker", 3, "common", _mult_if_contains(HandType.PAIR, 8))
_register("Zany Joker", 4, "common", _mult_if_contains(HandType.THREE_OF_A_KIND, 12))
_register("Mad Joker", 4, "common", _mult_if_contains(HandType.TWO_PAIR, 10))
_register("Crazy Joker", 4, "common", _mult_if_contains(HandType.STRAIGHT, 12))
_register("Droll Joker", 4, "common", _mult_if_contains(HandType.FLUSH, 10))
_register("Sly Joker", 3, "common", _chips_if_contains(HandType.PAIR, 50))
_register("Wily Joker", 4, "common", _chips_if_contains(HandType.THREE_OF_A_KIND, 100))
_register("Clever Joker", 4, "common", _chips_if_contains(HandType.TWO_PAIR, 80))
_register("Devious Joker", 4, "common", _chips_if_contains(HandType.STRAIGHT, 100))
_register("Crafty Joker", 4, "common", _chips_if_contains(HandType.FLUSH, 80))
_register("Half Joker", 5, "common", _mult_if_hand_size_le(3, 20))
_register("Banner", 5, "common", _chips_per_remaining_discard(30))
_register("Mystic Summit", 5, "common", _mult_if_no_discards(15))
_register("Even Steven", 4, "common", _mult_per_scoring_even_rank(4))
_register("Odd Todd", 4, "common", _chips_per_scoring_odd_rank(31))
_register("Scary Face", 4, "common", _chips_per_scoring_face(30))
_register("Abstract Joker", 4, "common", _mult_per_joker(3))


@dataclass
class Joker:
    spec: JokerSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def cost(self) -> int:
        return self.spec.cost

    @property
    def rarity(self) -> str:
        return self.spec.rarity

    def apply(self, chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return self.spec.effect(chips, mult, ctx)


def make_joker(name: str) -> Joker:
    return Joker(JOKER_CATALOG[name])
