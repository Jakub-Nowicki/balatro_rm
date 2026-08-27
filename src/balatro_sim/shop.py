from __future__ import annotations

import random
from dataclasses import dataclass

# Verified against balatrowiki.org/w/Shop.
CARD_SLOTS = 2
BASE_REROLL_COST = 5
REROLL_COST_INCREMENT = 1

# Price ranges by rarity, for use once a real joker catalog exists.
JOKER_PRICE_RANGE = {
    "common": (1, 6),
    "uncommon": (4, 8),
    "rare": (7, 10),
    "legendary": (20, 20),
}


@dataclass(frozen=True)
class ShopItem:
    name: str
    price: int
    rarity: str


class Shop:
    """Card-slot shop: rerolling and buying. Booster packs and vouchers are
    not modeled yet -- item_pool is expected to hold jokers/consumables once
    a real catalog exists (see data/jokers.json, still TODO)."""

    def __init__(self, item_pool: list[ShopItem], rng: random.Random | None = None):
        self.item_pool = item_pool
        self.rng = rng or random.Random()
        self.reroll_cost = BASE_REROLL_COST
        self.offerings: list[ShopItem] = []
        self._roll_offerings()

    def _roll_offerings(self) -> None:
        k = min(CARD_SLOTS, len(self.item_pool))
        self.offerings = self.rng.sample(self.item_pool, k) if k else []

    def reroll(self, money: int) -> int:
        if money < self.reroll_cost:
            raise ValueError("not enough money to reroll")
        cost = self.reroll_cost
        self._roll_offerings()
        self.reroll_cost += REROLL_COST_INCREMENT
        return cost

    def buy(self, item: ShopItem, money: int) -> int:
        if item not in self.offerings:
            raise ValueError(f"{item.name} is not currently offered")
        if money < item.price:
            raise ValueError("not enough money")
        self.offerings.remove(item)
        return item.price
