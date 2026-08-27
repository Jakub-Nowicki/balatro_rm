from __future__ import annotations

from enum import Enum


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

# Boss Blind special effects (debuffs, The Wall's 4x, The Needle's 1x, etc.)
# are not modeled yet -- every Boss Blind uses the generic 2x multiplier for now.
BLIND_MULTIPLIER = {Blind.SMALL: 1.0, Blind.BIG: 1.5, Blind.BOSS: 2.0}

BLIND_REWARD = {Blind.SMALL: 3, Blind.BIG: 4, Blind.BOSS: 5}


def blind_requirement(ante: int, blind: Blind) -> int:
    return int(ANTE_BASE_CHIPS[ante] * BLIND_MULTIPLIER[blind])


def blind_reward(blind: Blind) -> int:
    return BLIND_REWARD[blind]
