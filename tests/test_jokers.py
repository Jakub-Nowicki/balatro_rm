from balatro_sim.cards import Card, Rank, Suit
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.jokers import make_joker
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
