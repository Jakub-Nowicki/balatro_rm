from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from balatro_sim.cards import Card
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
) -> ScoreResult:
    base_chips, base_mult = BASE_SCORING[result.hand_type]
    card_chips = sum(c.rank.chip_value for c in result.scoring_cards)
    chips = base_chips + card_chips
    mult: float = base_mult

    played = played_cards if played_cards is not None else result.scoring_cards
    jokers = jokers or []

    if jokers:
        ctx = ScoringContext(
            played_cards=played,
            scoring_cards=result.scoring_cards,
            hand_type=result.hand_type,
            contained_hand_types=contained_hand_types(played),
            hands_remaining=hands_remaining,
            discards_remaining=discards_remaining,
            money=money,
            joker_count=len(jokers),
        )
        for joker in jokers:
            chips, mult = joker.apply(chips, mult, ctx)

    return ScoreResult(chips=chips, mult=mult, total=int(chips * mult))
