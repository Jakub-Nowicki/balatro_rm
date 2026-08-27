from __future__ import annotations

# Verified against balatrowiki.org/w/Money.
INTEREST_CHUNK = 5
MAX_INTEREST = 5
UNUSED_HAND_BONUS = 1


def calculate_interest(money: int) -> int:
    return min(MAX_INTEREST, money // INTEREST_CHUNK)
