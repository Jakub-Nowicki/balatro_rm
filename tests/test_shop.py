import random

from balatro_sim.shop import Shop, ShopItem


def make_pool(n: int) -> list[ShopItem]:
    return [ShopItem(name=f"item{i}", price=5, rarity="common") for i in range(n)]


def test_shop_offers_at_most_two_items():
    shop = Shop(make_pool(5), rng=random.Random(0))
    assert len(shop.offerings) == 2


def test_shop_offers_fewer_if_pool_smaller_than_slots():
    shop = Shop(make_pool(1), rng=random.Random(0))
    assert len(shop.offerings) == 1


def test_reroll_cost_starts_at_five_and_increments():
    shop = Shop(make_pool(5), rng=random.Random(0))
    assert shop.reroll_cost == 5
    shop.reroll(money=100)
    assert shop.reroll_cost == 6
    shop.reroll(money=100)
    assert shop.reroll_cost == 7


def test_reroll_raises_if_not_enough_money():
    shop = Shop(make_pool(5), rng=random.Random(0))
    try:
        shop.reroll(money=4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_buy_removes_item_from_offerings_and_returns_price():
    shop = Shop(make_pool(5), rng=random.Random(0))
    item = shop.offerings[0]
    price = shop.buy(item, money=100)
    assert price == item.price
    assert item not in shop.offerings


def test_buy_raises_if_item_not_offered():
    shop = Shop(make_pool(5), rng=random.Random(0))
    fake_item = ShopItem(name="not offered", price=1, rarity="common")
    try:
        shop.buy(fake_item, money=100)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_buy_raises_if_not_enough_money():
    shop = Shop(make_pool(5), rng=random.Random(0))
    item = shop.offerings[0]
    try:
        shop.buy(item, money=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_offerings_never_repeat_within_one_shop_visit():
    pool = make_pool(30)
    for seed in range(20):
        shop = Shop(pool, rng=random.Random(seed))
        names = [item.name for item in shop.offerings]
        assert len(names) == len(set(names))


def test_rarity_weighting_favors_common_over_uncommon_over_rare():
    mixed_pool = (
        [ShopItem(name=f"common{i}", price=2, rarity="common") for i in range(20)]
        + [ShopItem(name=f"uncommon{i}", price=5, rarity="uncommon") for i in range(20)]
        + [ShopItem(name=f"rare{i}", price=8, rarity="rare") for i in range(20)]
    )
    rng = random.Random(0)
    counts = {"common": 0, "uncommon": 0, "rare": 0}
    for _ in range(2000):
        shop = Shop(mixed_pool, rng=rng)
        for item in shop.offerings:
            counts[item.rarity] += 1

    # matches the verified 70/25/5 shop rarity weights, roughly
    assert counts["common"] > counts["uncommon"] > counts["rare"]
    total = sum(counts.values())
    assert 0.60 < counts["common"] / total < 0.80
    assert 0.15 < counts["uncommon"] / total < 0.35
