from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

from balatro_sim.cards import Card


class HandType(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    FIVE_OF_A_KIND = 9  # requires duplicate ranks via card enhancement, not detected yet
    FLUSH_HOUSE = 10  # requires duplicate ranks via card enhancement, not detected yet
    FLUSH_FIVE = 11  # requires duplicate ranks via card enhancement, not detected yet


# Level-1 base chips/mult per hand type, verified against balatrowiki.org/w/Poker_hands.
BASE_SCORING = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_OF_A_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_OF_A_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}


@dataclass
class HandResult:
    hand_type: HandType
    scoring_cards: list[Card]  # subset of played cards that count toward the hand


def _is_straight(ranks: list[int]) -> bool:
    unique = sorted(set(ranks))
    if len(unique) != 5:
        return False
    if unique[-1] - unique[0] == 4:
        return True
    # Ace-low straight: A,2,3,4,5
    return unique == [2, 3, 4, 5, 14]


def evaluate_hand(cards: list[Card]) -> HandResult:
    if not cards:
        raise ValueError("cannot evaluate an empty hand")

    ranks = [c.rank.value for c in cards]
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    is_flush = len(cards) == 5 and len(set(suits)) == 1
    is_straight = len(cards) == 5 and _is_straight(ranks)

    counts_desc = sorted(rank_counts.items(), key=lambda kv: (-kv[1], -kv[0]))

    def cards_for_ranks(target_ranks: set[int]) -> list[Card]:
        return [c for c in cards if c.rank.value in target_ranks]

    if is_straight and is_flush:
        return HandResult(HandType.STRAIGHT_FLUSH, list(cards))

    if counts_desc[0][1] == 4:
        quad_rank = counts_desc[0][0]
        return HandResult(HandType.FOUR_OF_A_KIND, cards_for_ranks({quad_rank}))

    if counts_desc[0][1] == 3 and len(counts_desc) > 1 and counts_desc[1][1] >= 2:
        trip_rank, pair_rank = counts_desc[0][0], counts_desc[1][0]
        return HandResult(HandType.FULL_HOUSE, cards_for_ranks({trip_rank, pair_rank}))

    if is_flush:
        return HandResult(HandType.FLUSH, list(cards))

    if is_straight:
        return HandResult(HandType.STRAIGHT, list(cards))

    if counts_desc[0][1] == 3:
        trip_rank = counts_desc[0][0]
        return HandResult(HandType.THREE_OF_A_KIND, cards_for_ranks({trip_rank}))

    pairs = [r for r, n in counts_desc if n == 2]
    if len(pairs) >= 2:
        top_two = pairs[:2]
        return HandResult(HandType.TWO_PAIR, cards_for_ranks(set(top_two)))

    if len(pairs) == 1:
        return HandResult(HandType.PAIR, cards_for_ranks({pairs[0]}))

    high_rank = max(ranks)
    return HandResult(HandType.HIGH_CARD, cards_for_ranks({high_rank}))


def contained_hand_types(cards: list[Card]) -> set[HandType]:
    """Every poker-hand pattern present in `cards`, not just the best one.

    Jokers like Jolly/Zany/Sly check for a pattern being present in the played
    cards (e.g. a Full House triggers both its Pair and Three of a Kind
    conditions), not just the single best-ranked hand type.
    """
    ranks = [c.rank.value for c in cards]
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    contained: set[HandType] = set()
    if counts[0] >= 2:
        contained.add(HandType.PAIR)
    if sum(1 for n in counts if n >= 2) >= 2:
        contained.add(HandType.TWO_PAIR)
    if counts[0] >= 3:
        contained.add(HandType.THREE_OF_A_KIND)
    if counts[0] >= 3 and len(counts) > 1 and counts[1] >= 2:
        contained.add(HandType.FULL_HOUSE)
    if counts[0] >= 4:
        contained.add(HandType.FOUR_OF_A_KIND)
    if len(cards) == 5 and len(set(suits)) == 1:
        contained.add(HandType.FLUSH)
    if len(cards) == 5 and _is_straight(ranks):
        contained.add(HandType.STRAIGHT)
    if HandType.STRAIGHT in contained and HandType.FLUSH in contained:
        contained.add(HandType.STRAIGHT_FLUSH)

    return contained
