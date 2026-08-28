import random

from balatro_sim.cards import Card, Rank, Suit
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.jokers import JOKER_CATALOG, make_joker, shop_item_pool
from balatro_sim.scoring import ScoringContext, score_hand


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def base_ctx(**overrides) -> ScoringContext:
    defaults = dict(
        played_cards=[],
        scoring_cards=[],
        hand_type=HandType.HIGH_CARD,
        contained_hand_types=set(),
        hands_remaining=0,
        discards_remaining=0,
        money=0,
        joker_count=1,
        joker_rarities=["common"],
        deck_size=52,
        max_joker_slots=5,
        rng=random.Random(0),
        all_cards_are_face=False,
    )
    defaults.update(overrides)
    return ScoringContext(**defaults)


def test_joker_flat_mult():
    joker = make_joker("Joker")
    chips, mult = joker.apply(10, 1, base_ctx())
    assert (chips, mult) == (10, 5)


def test_greedy_joker_counts_scoring_diamonds():
    joker = make_joker("Greedy Joker")
    scoring = [C(Rank.TWO, Suit.DIAMONDS), C(Rank.THREE, Suit.DIAMONDS), C(Rank.FOUR, Suit.HEARTS)]
    chips, mult = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert (chips, mult) == (10, 1 + 3 * 2)


def test_jolly_joker_triggers_on_contained_pair():
    joker = make_joker("Jolly Joker")
    chips, mult = joker.apply(10, 1, base_ctx(contained_hand_types={HandType.PAIR}))
    assert mult == 9
    chips, mult = joker.apply(10, 1, base_ctx(contained_hand_types=set()))
    assert mult == 1


def test_full_house_triggers_both_pair_and_trips_jokers():
    hand = [
        C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.CLUBS),
        C(Rank.TWO, Suit.DIAMONDS), C(Rank.TWO, Suit.SPADES),
    ]
    result = evaluate_hand(hand)
    assert result.hand_type == HandType.FULL_HOUSE

    jolly, zany = make_joker("Jolly Joker"), make_joker("Zany Joker")
    score = score_hand(result, played_cards=hand, jokers=[jolly, zany])
    # base full house: 40 + (7+7+7+2+2) = 65 chips, mult 4 -> +8 (Jolly, pair) -> +12 (Zany, trips) = 24
    assert score.chips == 65
    assert score.mult == 24
    assert score.total == 65 * 24


def test_sly_and_wily_add_chips_on_full_house():
    hand = [
        C(Rank.NINE, Suit.SPADES), C(Rank.NINE, Suit.HEARTS), C(Rank.NINE, Suit.CLUBS),
        C(Rank.FOUR, Suit.DIAMONDS), C(Rank.FOUR, Suit.SPADES),
    ]
    result = evaluate_hand(hand)
    sly, wily = make_joker("Sly Joker"), make_joker("Wily Joker")
    score = score_hand(result, played_cards=hand, jokers=[sly, wily])
    base_chips = 40 + (9 + 9 + 9 + 4 + 4)
    assert score.chips == base_chips + 50 + 100


def test_half_joker_only_triggers_on_small_hands():
    joker = make_joker("Half Joker")
    small_hand = [C(Rank.TWO, Suit.SPADES), C(Rank.THREE, Suit.HEARTS)]
    big_hand = [C(Rank.TWO, Suit.SPADES)] * 4
    _, mult_small = joker.apply(10, 1, base_ctx(played_cards=small_hand))
    _, mult_big = joker.apply(10, 1, base_ctx(played_cards=big_hand))
    assert mult_small == 21
    assert mult_big == 1


def test_banner_scales_with_remaining_discards():
    joker = make_joker("Banner")
    chips, _ = joker.apply(10, 1, base_ctx(discards_remaining=3))
    assert chips == 10 + 30 * 3


def test_mystic_summit_only_triggers_at_zero_discards():
    joker = make_joker("Mystic Summit")
    _, mult_zero = joker.apply(10, 1, base_ctx(discards_remaining=0))
    _, mult_some = joker.apply(10, 1, base_ctx(discards_remaining=1))
    assert mult_zero == 16
    assert mult_some == 1


def test_even_steven_counts_scoring_even_ranks():
    joker = make_joker("Even Steven")
    scoring = [C(Rank.FOUR, Suit.SPADES), C(Rank.SIX, Suit.HEARTS), C(Rank.FIVE, Suit.CLUBS)]
    _, mult = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert mult == 1 + 4 * 2


def test_odd_todd_counts_ace_as_odd():
    joker = make_joker("Odd Todd")
    scoring = [C(Rank.ACE, Suit.SPADES), C(Rank.THREE, Suit.HEARTS), C(Rank.FOUR, Suit.CLUBS)]
    chips, _ = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert chips == 10 + 31 * 2


def test_scary_face_counts_scoring_face_cards():
    joker = make_joker("Scary Face")
    scoring = [C(Rank.KING, Suit.SPADES), C(Rank.QUEEN, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS)]
    chips, _ = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert chips == 10 + 30 * 2


def test_abstract_joker_scales_with_joker_count():
    joker = make_joker("Abstract Joker")
    _, mult = joker.apply(10, 1, base_ctx(joker_count=3))
    assert mult == 1 + 3 * 3


def test_photograph_doubles_mult_on_first_scoring_face_card_only():
    joker = make_joker("Photograph")
    scoring = [C(Rank.KING, Suit.SPADES), C(Rank.QUEEN, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS)]
    _, mult = joker.apply(10, 3, base_ctx(scoring_cards=scoring))
    assert mult == 6  # x2 once, not once per face card

    no_face = [C(Rank.TWO, Suit.CLUBS), C(Rank.THREE, Suit.CLUBS)]
    _, mult_no_face = joker.apply(10, 3, base_ctx(scoring_cards=no_face))
    assert mult_no_face == 3


def test_baseball_card_scales_xmult_per_uncommon_joker():
    joker = make_joker("Baseball Card")
    _, mult = joker.apply(10, 4, base_ctx(joker_rarities=["common", "uncommon", "uncommon", "rare"]))
    assert mult == 4 * 1.5 * 1.5


def test_fibonacci_counts_scoring_cards_in_its_rank_set():
    joker = make_joker("Fibonacci")
    scoring = [C(Rank.ACE, Suit.SPADES), C(Rank.THREE, Suit.HEARTS), C(Rank.FOUR, Suit.CLUBS)]
    _, mult = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert mult == 1 + 8 * 2  # Ace and Three both count, Four does not


def test_shop_item_pool_defaults_to_every_joker():
    pool = shop_item_pool()
    assert len(pool) == len(JOKER_CATALOG)


def test_shop_item_pool_filters_by_rarity():
    pool = shop_item_pool({"common"})
    assert len(pool) > 0
    assert all(item.rarity == "common" for item in pool)
    assert len(pool) < len(JOKER_CATALOG)  # Fibonacci/Baseball Card are excluded


def test_blue_joker_scales_with_deck_size():
    joker = make_joker("Blue Joker")
    chips, _ = joker.apply(10, 1, base_ctx(deck_size=40))
    assert chips == 10 + 2 * 40


def test_cavendish_flat_xmult():
    joker = make_joker("Cavendish")
    _, mult = joker.apply(10, 2, base_ctx())
    assert mult == 6  # x3


def test_gros_michel_flat_mult():
    joker = make_joker("Gros Michel")
    _, mult = joker.apply(10, 1, base_ctx())
    assert mult == 16  # +15


def test_misprint_gives_random_mult_within_range():
    joker = make_joker("Misprint")
    rng = random.Random(42)
    results = [joker.apply(10, 1, base_ctx(rng=rng))[1] for _ in range(200)]
    assert all(1 <= m <= 1 + 23 for m in results)
    assert len(set(results)) > 1  # actually varies, not a constant


def test_scholar_gives_chips_and_mult_for_scoring_aces():
    joker = make_joker("Scholar")
    scoring = [C(Rank.ACE, Suit.SPADES), C(Rank.ACE, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS)]
    chips, mult = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert chips == 10 + 20 * 2
    assert mult == 1 + 4 * 2


def test_smiley_face_counts_scoring_face_cards():
    joker = make_joker("Smiley Face")
    scoring = [C(Rank.KING, Suit.SPADES), C(Rank.QUEEN, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS)]
    _, mult = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert mult == 1 + 5 * 2


def test_smiley_face_respects_pareidolia_all_cards_are_face():
    joker = make_joker("Smiley Face")
    scoring = [C(Rank.TWO, Suit.CLUBS), C(Rank.THREE, Suit.CLUBS)]
    _, mult = joker.apply(10, 1, base_ctx(scoring_cards=scoring, all_cards_are_face=True))
    assert mult == 1 + 5 * 2  # neither is really a face card, but Pareidolia makes them count


def test_square_joker_only_triggers_on_exactly_four_cards():
    joker = make_joker("Square Joker")
    four_cards = [C(Rank.TWO, Suit.CLUBS)] * 4
    five_cards = [C(Rank.TWO, Suit.CLUBS)] * 5
    chips_four, _ = joker.apply(10, 1, base_ctx(played_cards=four_cards))
    chips_five, _ = joker.apply(10, 1, base_ctx(played_cards=five_cards))
    assert chips_four == 14
    assert chips_five == 10


def test_walkie_talkie_counts_tens_and_fours():
    joker = make_joker("Walkie Talkie")
    scoring = [C(Rank.TEN, Suit.SPADES), C(Rank.FOUR, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS)]
    chips, mult = joker.apply(10, 1, base_ctx(scoring_cards=scoring))
    assert chips == 10 + 10 * 2
    assert mult == 1 + 4 * 2


def test_splash_makes_every_played_card_score_via_game_state():
    from balatro_sim.game_state import GameState

    game = GameState(rng=random.Random(0), jokers=[make_joker("Splash")])
    # a high-card hand where only the single highest card would normally score
    hand = [Card(Rank.KING, Suit.SPADES), Card(Rank.TWO, Suit.HEARTS), Card(Rank.THREE, Suit.CLUBS)]
    game.hand = hand + game.hand[len(hand):]
    score = game.play(hand)
    # base high card 5 + (10 King + 2 Two + 3 Three) = 20, instead of just 5+10=15 without Splash
    # (face cards are worth 10 chips in Balatro, not their face rank number)
    assert score.chips == 5 + 10 + 2 + 3


def test_bull_gives_chips_per_dollar_held():
    joker = make_joker("Bull")
    chips, _ = joker.apply(10, 1, base_ctx(money=15))
    assert chips == 10 + 2 * 15


def test_joker_stencil_scales_xmult_with_empty_slots():
    joker = make_joker("Joker Stencil")
    _, mult_with_room = joker.apply(10, 2, base_ctx(joker_count=2, max_joker_slots=5))
    _, mult_full = joker.apply(10, 2, base_ctx(joker_count=5, max_joker_slots=5))
    assert mult_with_room == 2 * 4  # 3 empty slots + 1 (itself) = X4
    assert mult_full == 2  # no empty slots -> X1, unchanged
