from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from balatro_sim.cards import Suit


class Blind(Enum):
    SMALL = "Small Blind"
    BIG = "Big Blind"
    BOSS = "Boss Blind"


BLIND_ORDER = [Blind.SMALL, Blind.BIG, Blind.BOSS]

# Verified against balatrowiki.org/w/Blinds. Ante 9+ (endless mode) not modeled.
ANTE_BASE_CHIPS = {
    1: 300,
    2: 800,
    3: 2_000,
    4: 5_000,
    5: 11_000,
    6: 20_000,
    7: 35_000,
    8: 50_000,
}

# None of the 8 bosses eligible at Ante 1 override the requirement multiplier
# (The Wall's 4x, The Needle's 1x etc. only appear at higher antes), so 2x is
# accurate for every Boss Blind this simulator can currently produce.
BLIND_MULTIPLIER = {Blind.SMALL: 1.0, Blind.BIG: 1.5, Blind.BOSS: 2.0}

BLIND_REWARD = {Blind.SMALL: 3, Blind.BIG: 4, Blind.BOSS: 5}


def blind_requirement(ante: int, blind: Blind) -> int:
    return int(ANTE_BASE_CHIPS[ante] * BLIND_MULTIPLIER[blind])


def blind_reward(blind: Blind) -> int:
    return BLIND_REWARD[blind]


@dataclass(frozen=True)
class BossBlind:
    name: str
    debuffed_suit: Suit | None = None
    hand_size_delta: int = 0
    auto_discard_after_play: int = 0
    required_play_size: int | None = None
    debuffs_previously_played_cards: bool = False


# All 8 bosses that can appear at Ante 1, verified against
# balatrowiki.org/w/Blinds_and_Antes.
IMPLEMENTED_ANTE_1_BOSSES = [
    BossBlind("The Club", debuffed_suit=Suit.CLUBS),
    BossBlind("The Goad", debuffed_suit=Suit.SPADES),
    BossBlind("The Window", debuffed_suit=Suit.DIAMONDS),
    BossBlind("The Head", debuffed_suit=Suit.HEARTS),
    BossBlind("The Manacle", hand_size_delta=-1),
    BossBlind("The Hook", auto_discard_after_play=2),
    BossBlind("The Psychic", required_play_size=5),
    BossBlind("The Pillar", debuffs_previously_played_cards=True),
]
