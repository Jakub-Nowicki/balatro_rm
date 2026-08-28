import random

from balatro_sim.blinds import Blind, BossBlind
from balatro_sim.cards import Card, Rank, Suit
from balatro_sim.game_state import GameState
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.jokers import make_joker
from balatro_sim.scoring import score_hand


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def test_debuffed_suit_contributes_zero_chips():
    hand = [C(Rank.KING, Suit.HEARTS), C(Rank.KING, Suit.CLUBS)]
    result = evaluate_hand(hand)
    normal = score_hand(result)
    debuffed = score_hand(result, debuffed_suit=Suit.HEARTS)
    # base 10 + (10 King + 10 King) = 30 normally; with Hearts debuffed, only the Club King counts
    assert normal.chips == 30
    assert debuffed.chips == 20


def test_debuffed_suit_still_counts_toward_hand_type():
    hand = [C(Rank.KING, Suit.HEARTS), C(Rank.KING, Suit.CLUBS)]
    result = evaluate_hand(hand)
    # a Pair of debuffed-suit Kings is still detected as a Pair -- the debuff
    # only zeroes the card's own chip contribution, not hand-type detection.
    assert result.hand_type == HandType.PAIR
    debuffed = score_hand(result, debuffed_suit=Suit.HEARTS)
    assert debuffed.total > 0


def test_debuffed_suit_blocks_matching_suit_joker():
    hand = [C(Rank.TWO, Suit.DIAMONDS), C(Rank.THREE, Suit.DIAMONDS)]
    result = evaluate_hand(hand)
    greedy = make_joker("Greedy Joker")  # +3 Mult per scoring Diamond
    normal = score_hand(result, played_cards=hand, jokers=[greedy])
    debuffed = score_hand(result, played_cards=hand, jokers=[greedy], debuffed_suit=Suit.DIAMONDS)
    assert normal.mult > 1  # Greedy Joker triggered
    assert debuffed.mult == 1  # blocked -- no scoring Diamonds left to trigger it


def test_gamestate_assigns_active_boss_only_on_boss_blind():
    game = GameState(rng=random.Random(0))
    assert game.active_boss is None  # starts on Small Blind

    game.round_chips = game.requirement
    game.collect_reward_and_advance()
    assert game.blind == Blind.BIG
    assert game.active_boss is None  # Big Blind has no boss effect

    game.round_chips = game.requirement
    game.collect_reward_and_advance()
    assert game.blind == Blind.BOSS
    assert game.active_boss is not None
    game.leave_shop()

    game.round_chips = game.requirement
    game.collect_reward_and_advance()
    assert game.blind == Blind.SMALL
    assert game.ante == 2
    assert game.active_boss is None  # boss effect cleared after the Boss round ends


def test_manacle_reduces_hand_size_for_the_round():
    manacle = BossBlind("The Manacle", hand_size_delta=-1)
    game = GameState(blind=Blind.BOSS, rng=random.Random(0))
    game.active_boss = manacle
    game._start_round()
    assert len(game.hand) == game.hand_size - 1


def test_suit_debuff_boss_reduces_actual_round_chips_via_play():
    club_boss = BossBlind("The Club", debuffed_suit=Suit.CLUBS)
    game = GameState(blind=Blind.BOSS, rng=random.Random(0))
    game.active_boss = club_boss
    pair_of_kings = [Card(Rank.KING, Suit.CLUBS), Card(Rank.KING, Suit.HEARTS)]
    game.hand = pair_of_kings + game.hand[len(pair_of_kings):]

    score = game.play(pair_of_kings)
    # base pair: 10 + (0 debuffed Club King + 10 Heart King) = 20 chips, mult 2
    assert score.chips == 20
    assert score.total == 20 * 2


def test_the_hook_auto_discards_two_random_cards_after_play_for_free():
    hook = BossBlind("The Hook", auto_discard_after_play=2)
    game = GameState(blind=Blind.BOSS, rng=random.Random(0))
    game.active_boss = hook
    discards_before = game.discards_remaining
    deck_size_before = len(game.deck)

    game.play(game.hand[:2])

    # hand refilled back to full, discards_remaining untouched (the auto-
    # discard is free), and the deck lost 4 cards: 2 to refill after the
    # play itself, 2 more to refill after the boss's free auto-discard.
    assert len(game.hand) == game._effective_hand_size()
    assert game.discards_remaining == discards_before
    assert len(game.deck) == deck_size_before - 4


def test_the_hook_skips_auto_discard_if_round_already_over():
    hook = BossBlind("The Hook", auto_discard_after_play=2)
    game = GameState(blind=Blind.BOSS, rng=random.Random(0))
    game.active_boss = hook
    game.hands_remaining = 1  # this play will end the round regardless of score
    game.round_chips = game.requirement  # already beaten, so is_round_over is True post-play
    deck_size_before = len(game.deck)

    game.play(game.hand[:1])
    # round is over -- no auto-discard should have fired, so the deck only
    # lost 1 card (the normal refill draw after playing 1 card), not 3
    # (that refill + the boss's free 2-card auto-discard refill).
    assert game.hands_remaining == 0
    assert len(game.deck) == deck_size_before - 1


def test_the_psychic_requires_exactly_five_cards():
    psychic = BossBlind("The Psychic", required_play_size=5)
    game = GameState(blind=Blind.BOSS, rng=random.Random(0))
    game.active_boss = psychic

    try:
        game.play(game.hand[:3])
        assert False, "expected ValueError for a non-5-card play under The Psychic"
    except ValueError:
        pass

    score = game.play(game.hand[:5])
    assert score.total > 0


def test_the_pillar_debuffs_cards_played_earlier_this_ante():
    pillar = BossBlind("The Pillar", debuffs_previously_played_cards=True)
    game = GameState(blind=Blind.BOSS, rng=random.Random(0))
    game.active_boss = pillar
    king_hearts = Card(Rank.KING, Suit.HEARTS)
    filler = [c for c in game.hand if c != king_hearts][:4]
    game.hand = [king_hearts] + filler + game.hand[5:]

    first = game.play([king_hearts])
    assert first.chips == 5 + 10  # base high card 5 + King's 10 chips, not yet debuffed

    # put King of Hearts back in hand and play it again this same ante
    game.hand = [king_hearts] + game.hand
    second = game.play([king_hearts])
    # same base high-card scoring, but the King now contributes 0 chips
    # since it was already played earlier this ante
    assert second.chips == 5
