from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from balatro_sim.cards import Rank, Suit
from balatro_sim.hands import HandType
from balatro_sim.scoring import ScoringContext
from balatro_sim.shop import ShopItem

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


def _is_face(card, ctx: ScoringContext) -> bool:
    # Pareidolia ("all cards are considered face cards") sets ctx.all_cards_are_face.
    return card.rank in FACE_RANKS or ctx.all_cards_are_face


def _chips_per_scoring_face(amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if _is_face(c, ctx))
        return chips + amount * n, mult
    return fn


def _mult_per_joker(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips, mult + amount * ctx.joker_count
    return fn


def _xmult_on_first_scoring_face_card(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        for c in ctx.scoring_cards:
            if c.rank in FACE_RANKS:
                return chips, mult * amount
        return chips, mult
    return fn


def _xmult_per_uncommon_joker(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for r in ctx.joker_rarities if r == "uncommon")
        return chips, mult * (amount**n)
    return fn


def _mult_per_scoring_rank_in(values: set[int], amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if c.rank.value in values)
        return chips, mult + amount * n
    return fn


def _flat_xmult(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips, mult * amount
    return fn


def _chips_per_deck_card(amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips + amount * ctx.deck_size, mult
    return fn


def _random_mult(low: float, high: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips, mult + ctx.rng.uniform(low, high)
    return fn


def _chips_and_mult_per_scoring_rank_in(values: set[int], chips_amount: int, mult_amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if c.rank.value in values)
        return chips + chips_amount * n, mult + mult_amount * n
    return fn


def _mult_per_scoring_face(amount: float) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        n = sum(1 for c in ctx.scoring_cards if _is_face(c, ctx))
        return chips, mult + amount * n
    return fn


def _chips_if_hand_size_eq(n_cards: int, amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        if len(ctx.played_cards) == n_cards:
            return chips + amount, mult
        return chips, mult
    return fn


def _chips_per_dollar(amount: int) -> EffectFn:
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        return chips + amount * max(0, ctx.money), mult
    return fn


def _xmult_per_empty_joker_slot() -> EffectFn:
    # "X1 Mult for each empty Joker slot, Joker Stencil included" -- the total
    # multiplier is (empty_slots + 1): the "+1" is Joker Stencil counting
    # itself. A per-slot X1 compounded via repeated multiplication would be a
    # mathematical no-op (1 * 1 * 1... == 1), which can't be the real effect
    # for a purchasable joker, so this reads it as a single multiplication.
    def fn(chips: int, mult: float, ctx: ScoringContext) -> tuple[int, float]:
        empty_slots = max(0, ctx.max_joker_slots - ctx.joker_count)
        return chips, mult * (empty_slots + 1)
    return fn


@dataclass(frozen=True)
class JokerSpec:
    name: str
    cost: int
    rarity: str
    effect: EffectFn
    all_cards_score: bool = False
    all_cards_are_face: bool = False


# Verified against balatrowiki.org/w/Jokers. A curated subset spanning the main
# effect shapes (flat mult/chips, per-suit, hand-pattern-conditional, positional,
# resource-conditional) rather than the full 150-joker roster.
JOKER_CATALOG: dict[str, JokerSpec] = {}


def _register(
    name: str,
    cost: int,
    rarity: str,
    effect: EffectFn,
    all_cards_score: bool = False,
    all_cards_are_face: bool = False,
) -> None:
    JOKER_CATALOG[name] = JokerSpec(name, cost, rarity, effect, all_cards_score, all_cards_are_face)


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
_register("Photograph", 5, "common", _xmult_on_first_scoring_face_card(2.0))
_register("Baseball Card", 8, "rare", _xmult_per_uncommon_joker(1.5))
_register("Fibonacci", 8, "uncommon", _mult_per_scoring_rank_in({14, 2, 3, 5, 8}, 8))
_register("Abstract Joker", 4, "common", _mult_per_joker(3))

# Second batch, verified against balatrowiki.org/w/Jokers. Of the ~74 jokers
# surveyed for this batch (all remaining unlocked-from-start common and
# uncommon jokers), only these 12 fit the current stateless per-hand scoring
# architecture. The rest need systems not built yet: persistent per-joker
# state that accumulates/decays across hands or rounds (e.g. Green Joker,
# Ride the Bus, Hiker, Loyalty Card), a money-granting hook tied to round-end
# or blind-selection events rather than scoring (Golden Joker, Egg, Rocket),
# Tarot/Planet/Spectral card dependencies (8 Ball, Fortune Teller, Space
# Joker, Seance), card enhancements/deck composition (Steel Joker, Vampire,
# Midas Mask), joker-selling (Diet Cola, Luchador), or hand-evaluation rule
# changes beyond what Splash/Pareidolia need (Four Fingers, Shortcut).
# Two known simplifications below: Cavendish and Gros Michel each have a
# small (1/1000, 1/6) per-round chance of being destroyed in the real game --
# omitted since joker removal isn't implemented; both keep their scoring effect.
_register("Blue Joker", 5, "common", _chips_per_deck_card(2))
_register("Cavendish", 4, "common", _flat_xmult(3.0))
_register("Gros Michel", 5, "common", _flat_mult(15))
_register("Misprint", 4, "common", _random_mult(0, 23))
_register("Scholar", 4, "common", _chips_and_mult_per_scoring_rank_in({14}, 20, 4))
_register("Smiley Face", 4, "common", _mult_per_scoring_face(5))
_register("Square Joker", 4, "common", _chips_if_hand_size_eq(4, 4))
_register("Walkie Talkie", 4, "common", _chips_and_mult_per_scoring_rank_in({10, 4}, 10, 4))
_register("Splash", 3, "common", lambda chips, mult, ctx: (chips, mult), all_cards_score=True)
_register("Bull", 6, "uncommon", _chips_per_dollar(2))
_register("Joker Stencil", 8, "uncommon", _xmult_per_empty_joker_slot())
_register("Pareidolia", 5, "uncommon", lambda chips, mult, ctx: (chips, mult), all_cards_are_face=True)


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


# Stable ordering for observation-vector encoding of "which joker is this".
JOKER_NAMES: list[str] = list(JOKER_CATALOG.keys())
JOKER_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOKER_NAMES)}


def shop_item_pool(rarities: set[str] | None = None) -> list[ShopItem]:
    specs = JOKER_CATALOG.values() if rarities is None else (s for s in JOKER_CATALOG.values() if s.rarity in rarities)
    return [ShopItem(name=spec.name, price=spec.cost, rarity=spec.rarity) for spec in specs]
