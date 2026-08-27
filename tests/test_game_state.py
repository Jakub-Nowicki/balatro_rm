import random

from balatro_sim.blinds import Blind, blind_requirement
from balatro_sim.cards import Card, Rank, Suit
from balatro_sim.game_state import GameState
from balatro_sim.jokers import make_joker


def test_blind_requirements():
    assert blind_requirement(1, Blind.SMALL) == 300
    assert blind_requirement(1, Blind.BIG) == 450
    assert blind_requirement(1, Blind.BOSS) == 600
    assert blind_requirement(8, Blind.SMALL) == 50_000


def test_new_game_deals_full_hand():
    game = GameState(rng=random.Random(0))
    assert len(game.hand) == game.hand_size
    assert game.hands_remaining == game.hands_per_round
    assert game.discards_remaining == game.discards_per_round
    assert game.round_chips == 0


def test_play_hand_scores_and_redraws():
    game = GameState(rng=random.Random(0))
    initial_hands = game.hands_remaining
    played = game.hand[:2]
    score = game.play(played)

    assert game.hands_remaining == initial_hands - 1
    assert game.round_chips == score.total
    assert len(game.hand) == game.hand_size
    assert all(c not in game.hand for c in played)


def test_discard_redraws_without_scoring():
    game = GameState(rng=random.Random(0))
    initial_discards = game.discards_remaining
    discarded = game.hand[:3]
    game.discard(discarded)

    assert game.discards_remaining == initial_discards - 1
    assert game.round_chips == 0
    assert len(game.hand) == game.hand_size
    assert all(c not in game.hand for c in discarded)


def test_playing_more_hands_than_remaining_raises():
    game = GameState(rng=random.Random(0), hands_per_round=1)
    game.play(game.hand[:1])
    assert game.hands_remaining == 0
    try:
        game.play(game.hand[:1])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_beat_blind_and_advance_to_big_blind():
    game = GameState(rng=random.Random(0))
    game.round_chips = game.requirement  # force a win regardless of card luck
    assert game.is_blind_beaten

    starting_money = game.money
    result = game.collect_reward_and_advance()

    # 4 starting money -> $0 interest; all 4 hands unused -> $4 bonus
    assert result.blind_reward == 3
    assert result.hand_bonus == 4
    assert result.interest == 0
    assert result.total == 7
    assert game.money == starting_money + 7
    assert game.blind == Blind.BIG
    assert game.ante == 1
    assert game.round_chips == 0
    assert game.hands_remaining == game.hands_per_round


def test_cash_out_interest_scales_with_money_held_and_caps_at_5():
    game = GameState(rng=random.Random(0), starting_money=27)
    game.round_chips = game.requirement
    result = game.collect_reward_and_advance()
    assert result.interest == 5  # 27 // 5 = 5, already at the cap


def test_cash_out_hand_bonus_scales_with_unused_hands():
    game = GameState(rng=random.Random(0))
    game.play(game.hand[:1])  # burn one hand, 3 remain unused
    game.round_chips = game.requirement
    result = game.collect_reward_and_advance()
    assert result.hand_bonus == 3


def test_beat_boss_blind_advances_ante():
    game = GameState(blind=Blind.BOSS, rng=random.Random(0))
    game.round_chips = game.requirement
    game.collect_reward_and_advance()

    assert game.blind == Blind.SMALL
    assert game.ante == 2


def test_loss_when_hands_exhausted_without_beating_requirement():
    game = GameState(rng=random.Random(0), hands_per_round=1)
    game.play(game.hand[:1])  # a single random 1-card hand won't reach 300 chips
    assert game.is_game_over_loss
    assert not game.is_blind_beaten


def test_play_applies_equipped_jokers():
    jokers = [make_joker("Joker"), make_joker("Jolly Joker")]
    game = GameState(rng=random.Random(0), jokers=jokers)
    pair_of_kings = [Card(Rank.KING, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    game.hand = pair_of_kings + game.hand[len(pair_of_kings):]

    score = game.play(pair_of_kings)
    # base pair: 10 + (10+10) = 30 chips, mult 2 -> +4 (Joker) -> +8 (Jolly, pair) = 14
    assert score.chips == 30
    assert score.mult == 14
    assert score.total == 30 * 14
    assert game.round_chips == 30 * 14
