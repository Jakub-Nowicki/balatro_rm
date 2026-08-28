from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from balatro_sim.cards import Card, Suit
from balatro_sim.hands import BASE_SCORING, HandResult, HandType, contained_hand_types

if TYPE_CHECKING:
    from balatro_sim.jokers import Joker


@dataclass
class ScoringContext:
    played_cards: list[Card]
    scoring_cards: list[Card]
    hand_type: HandType
    contained_hand_types: set[HandType]
    hands_remaining: int
    discards_remaining: int
    money: int
    joker_count: int
    joker_rarities: list[str]
    deck_size: int
    max_joker_slots: int
    rng: random.Random
    all_cards_are_face: bool


@dataclass
class ScoreResult:
    chips: int
    mult: float
    total: int


def score_hand(
    result: HandResult,
    played_cards: list[Card] | None = None,
    jokers: "list[Joker] | None" = None,
    hands_remaining: int = 0,
    discards_remaining: int = 0,
    money: int = 0,
    debuffed_suit: Suit | None = None,
    debuffed_cards: "set[Card] | None" = None,
    deck_size: int = 0,
    max_joker_slots: int = 0,
    rng: random.Random | None = None,
) -> ScoreResult:
    jokers = jokers or []
    played = played_cards if played_cards is not None else result.scoring_cards
    debuffed_cards = debuffed_cards or set()

    # Splash ("every played card counts in scoring") replaces the hand's
    # normal scoring-card subset (e.g. just the pair, for a Pair) with every
    # played card -- a rule change, not an additive effect, so it's applied
    # before the rest of scoring rather than through the usual joker hook.
    scoring_source = played if any(j.spec.all_cards_score for j in jokers) else result.scoring_cards

    # A debuffed card (e.g. a suit-debuffing Boss Blind, or a specific card
    # The Pillar debuffs for having been played earlier this ante) still
    # counts toward hand-type detection -- you can still make a flush out of
    # debuffed hearts -- but contributes 0 chips and can't trigger
    # suit/rank-matching joker effects, matching the wiki's note that Greedy
    # Joker etc. stop working against their matching debuffed suit.
    effective_scoring_cards = [
        c for c in scoring_source if c.suit != debuffed_suit and c not in debuffed_cards
    ]

    base_chips, base_mult = BASE_SCORING[result.hand_type]
    card_chips = sum(c.rank.chip_value for c in effective_scoring_cards)
    chips = base_chips + card_chips
    mult: float = base_mult

    if jokers:
        ctx = ScoringContext(
            played_cards=played,
            scoring_cards=effective_scoring_cards,
            hand_type=result.hand_type,
            contained_hand_types=contained_hand_types(played),
            hands_remaining=hands_remaining,
            discards_remaining=discards_remaining,
            money=money,
            joker_count=len(jokers),
            joker_rarities=[j.rarity for j in jokers],
            deck_size=deck_size,
            max_joker_slots=max_joker_slots,
            rng=rng or random.Random(),
            all_cards_are_face=any(j.spec.all_cards_are_face for j in jokers),
        )
        for joker in jokers:
            chips, mult = joker.apply(chips, mult, ctx)

    return ScoreResult(chips=chips, mult=mult, total=int(chips * mult))
