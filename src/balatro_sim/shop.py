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

# Verified against balatrowiki.org/community sources: when the shop generates
# a card slot, it picks a rarity tier with these weights, then a specific
# joker within that tier. The wiki doesn't document the within-tier
# selection rule explicitly; uniform-within-tier is the standard community
# understanding, not independently wiki-verified.
RARITY_WEIGHTS = {"common": 70, "uncommon": 25, "rare": 5}


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
        # Picks a rarity tier per slot (weighted by RARITY_WEIGHTS, renormalized
        # over whichever rarities are actually present in the pool -- e.g. an
        # Ante-1 common-only pool always resolves to "common"), then a specific
        # joker uniformly within that tier, without replacement across slots
        # in the same shop visit.
        self.offerings = []
        available = list(self.item_pool)
        for _ in range(min(CARD_SLOTS, len(available))):
            rarities_present = sorted({item.rarity for item in available})
            weights = [RARITY_WEIGHTS.get(r, 0) for r in rarities_present]
            if sum(weights) == 0:
                chosen_rarity = self.rng.choice(rarities_present)
            else:
                chosen_rarity = self.rng.choices(rarities_present, weights=weights, k=1)[0]
            candidates = [item for item in available if item.rarity == chosen_rarity]
            chosen_item = self.rng.choice(candidates)
            self.offerings.append(chosen_item)
            available.remove(chosen_item)

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
