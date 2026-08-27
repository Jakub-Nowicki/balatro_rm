from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random


class Suit(Enum):
    SPADES = "S"
    HEARTS = "H"
    CLUBS = "C"
    DIAMONDS = "D"


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def chip_value(self) -> int:
        if self is Rank.ACE:
            return 11
        if self.value >= 10:
            return 10
        return self.value


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __repr__(self) -> str:
        return f"{self.rank.name}{self.suit.value}"


def standard_deck() -> list[Card]:
    return [Card(rank, suit) for suit in Suit for rank in Rank]


class Deck:
    def __init__(self, cards: list[Card] | None = None):
        self.cards = cards if cards is not None else standard_deck()

    def shuffle(self, rng: random.Random | None = None) -> None:
        (rng or random).shuffle(self.cards)

    def draw(self, n: int) -> list[Card]:
        drawn, self.cards = self.cards[:n], self.cards[n:]
        return drawn

    def __len__(self) -> int:
        return len(self.cards)
