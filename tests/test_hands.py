from balatro_sim.cards import Card, Rank, Suit
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.scoring import score_hand


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def test_high_card():
    hand = [C(Rank.TWO, Suit.SPADES), C(Rank.FIVE, Suit.HEARTS), C(Rank.NINE, Suit.CLUBS)]
    result = evaluate_hand(hand)
    assert result.hand_type == HandType.HIGH_CARD
    assert result.scoring_cards == [C(Rank.NINE, Suit.CLUBS)]


def test_pair():
    hand = [C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS)]
    result = evaluate_hand(hand)
    assert result.hand_type == HandType.PAIR
    assert len(result.scoring_cards) == 2


def test_two_pair():
    hand = [
        C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
        C(Rank.TWO, Suit.CLUBS), C(Rank.TWO, Suit.DIAMONDS),
        C(Rank.FIVE, Suit.SPADES),
    ]
    result = evaluate_hand(hand)
    assert result.hand_type == HandType.TWO_PAIR
    assert len(result.scoring_cards) == 4


def test_three_of_a_kind():
    hand = [C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.CLUBS)]
    assert evaluate_hand(hand).hand_type == HandType.THREE_OF_A_KIND


def test_straight():
    hand = [
        C(Rank.FOUR, Suit.SPADES), C(Rank.FIVE, Suit.HEARTS), C(Rank.SIX, Suit.CLUBS),
        C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.EIGHT, Suit.SPADES),
    ]
    assert evaluate_hand(hand).hand_type == HandType.STRAIGHT


def test_ace_low_straight():
    hand = [
        C(Rank.ACE, Suit.SPADES), C(Rank.TWO, Suit.HEARTS), C(Rank.THREE, Suit.CLUBS),
        C(Rank.FOUR, Suit.DIAMONDS), C(Rank.FIVE, Suit.SPADES),
    ]
    assert evaluate_hand(hand).hand_type == HandType.STRAIGHT


def test_flush():
    hand = [
        C(Rank.TWO, Suit.SPADES), C(Rank.FIVE, Suit.SPADES), C(Rank.NINE, Suit.SPADES),
        C(Rank.JACK, Suit.SPADES), C(Rank.KING, Suit.SPADES),
    ]
    assert evaluate_hand(hand).hand_type == HandType.FLUSH


def test_full_house():
    hand = [
        C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.CLUBS),
        C(Rank.TWO, Suit.DIAMONDS), C(Rank.TWO, Suit.SPADES),
    ]
    assert evaluate_hand(hand).hand_type == HandType.FULL_HOUSE


def test_four_of_a_kind():
    hand = [
        C(Rank.NINE, Suit.SPADES), C(Rank.NINE, Suit.HEARTS),
        C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS),
    ]
    assert evaluate_hand(hand).hand_type == HandType.FOUR_OF_A_KIND


def test_straight_flush():
    hand = [
        C(Rank.FOUR, Suit.SPADES), C(Rank.FIVE, Suit.SPADES), C(Rank.SIX, Suit.SPADES),
        C(Rank.SEVEN, Suit.SPADES), C(Rank.EIGHT, Suit.SPADES),
    ]
    assert evaluate_hand(hand).hand_type == HandType.STRAIGHT_FLUSH


def test_score_pair_of_kings():
    hand = [C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS)]
    result = evaluate_hand(hand)
    score = score_hand(result)
    # base 10 chips + (10+10 for the two kings) = 30 chips, x2 mult = 60
    assert score.chips == 30
    assert score.mult == 2
    assert score.total == 60
