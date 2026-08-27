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
