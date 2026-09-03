import numpy as np

from balatro_sim.blinds import IMPLEMENTED_ANTE_1_BOSSES, Blind, BossBlind
from balatro_sim.env import (
    CARD_FEATURES,
    DISCARD_COST,
    HAND_SLOTS,
    INVALID_ACTION_PENALTY,
    MAX_ACHIEVABLE_HAND_TYPE,
    OBS_SIZE,
    SHOP_SLOTS,
    BalatroEnv,
)
from balatro_sim.hands import HandType, evaluate_hand

# Index of the best-achievable-hand-type feature: right after the 8 hand
# slots x CARD_FEATURES block and the 7 scalars that follow it.
BEST_HAND_TYPE_OBS_INDEX = HAND_SLOTS * CARD_FEATURES + 7
# The 8 per-slot "is this card part of the best achievable combo" flags
# immediately follow the scalar hand-type feature.
COMBO_MEMBERSHIP_OBS_START = BEST_HAND_TYPE_OBS_INDEX + 1


def _round_action(subset: tuple[int, ...], mode: int) -> np.ndarray:
    flags = [1 if i in subset else 0 for i in range(HAND_SLOTS)]
    return np.array(flags + [mode, 0], dtype=int)


def _shop_action(choice: int) -> np.ndarray:
    return np.array([0] * HAND_SLOTS + [0, choice], dtype=int)


def test_reset_returns_correctly_shaped_obs():
    env = BalatroEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert info == {}


def test_action_space_and_observation_space_shapes():
    env = BalatroEnv()
    assert env.action_space.shape == (HAND_SLOTS + 2,)
    assert env.observation_space.shape == (OBS_SIZE,)


def test_round_action_with_no_cards_flagged_falls_back_to_a_single_card_play():
    env = BalatroEnv()
    env.reset(seed=0)
    action = np.array([0] * HAND_SLOTS + [0, 0], dtype=int)  # nothing flagged, mode=play
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward != INVALID_ACTION_PENALTY
    assert env.game.hands_remaining == env.game.hands_per_round - 1  # a play actually happened


def test_round_action_with_more_than_five_cards_flagged_is_truncated_to_five():
    env = BalatroEnv()
    env.reset(seed=0)
    action = np.array([1] * HAND_SLOTS + [0, 0], dtype=int)  # all 8 flagged, mode=play
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward != INVALID_ACTION_PENALTY
    assert env.game.hands_remaining == env.game.hands_per_round - 1


def test_valid_play_scores_and_advances_state():
    env = BalatroEnv()
    env.reset(seed=0)
    action = _round_action((0,), mode=0)  # play hand slot 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.game.hands_remaining == env.game.hands_per_round - 1


def test_valid_discard_incurs_small_cost():
    env = BalatroEnv()
    env.reset(seed=0)
    action = _round_action((0,), mode=1)  # discard hand slot 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward == DISCARD_COST
    assert env.game.discards_remaining == env.game.discards_per_round - 1


def test_discard_with_zero_discards_falls_back_to_play_instead_of_soft_locking():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.discards_remaining = 0
    action = _round_action((0,), mode=1)  # request discard, but none are left
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.game.hands_remaining == env.game.hands_per_round - 1  # treated as a play, not invalid


def test_episode_terminates_with_random_actions():
    env = BalatroEnv()
    env.reset(seed=42)
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert info.get("result") in ("lost", "won_run")
            return
    raise AssertionError("episode did not terminate within 200 random steps")


def test_shop_phase_buy_adds_joker_and_deducts_money():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.round_chips = env.game.requirement  # force the blind beaten
    env.step(_round_action((0,), mode=0))  # this play beats the blind and enters the shop phase
    assert env.game.phase == "shop"

    env.game.money = 100  # guarantee affordability regardless of the offering rolled
    item = env.game.shop.offerings[0]
    starting_jokers = len(env.game.jokers)

    shop_action = _shop_action(0)  # buy offering[0]
    env.step(shop_action)

    assert len(env.game.jokers) == starting_jokers + 1
    assert env.game.money == 100 - item.price


def test_shop_phase_leave_returns_to_round_phase():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.round_chips = env.game.requirement
    env.step(_round_action((0,), mode=0))
    assert env.game.phase == "shop"

    leave_action = _shop_action(SHOP_SLOTS + 1)
    env.step(leave_action)
    assert env.game.phase == "round"
    assert env.game.hands_remaining == env.game.hands_per_round


def test_enable_shop_false_skips_shop_transparently():
    env = BalatroEnv(enable_shop=False)
    env.reset(seed=0)
    env.game.round_chips = env.game.requirement
    obs, reward, terminated, truncated, info = env.step(_round_action((0,), mode=0))
    assert env.game.phase == "round"  # shop was auto-skipped within the same step
    assert env.game.hands_remaining == env.game.hands_per_round


def test_flagged_slot_beyond_a_shrunk_hand_still_plays_the_cards_that_exist():
    # Regression test: under a hand-size-reducing boss (The Manacle), a
    # flagged slot that references the now-missing card used to be rejected
    # as invalid outright -- if a deterministic policy kept re-flagging that
    # same slot (same observation, same argmax action every step, nothing
    # changes to break the cycle), it would never make progress, running out
    # the clock at the environment's step cap. Confirmed happening during a
    # real overnight training run once the agent got skilled enough to
    # regularly reach Boss Blind rounds.
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.blind = Blind.BOSS
    env.game.active_boss = BossBlind("The Manacle", hand_size_delta=-1)
    env.game._start_round()
    assert len(env.game.hand) == 7

    # flags slot 0 (valid) and slot 7 (now out of range) -- needs a partial
    # overlap so slot 0 alone doesn't collapse into the same case as the
    # "nothing valid selected" fallback tested separately above.
    action = _round_action((0, 7), mode=0)
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward != INVALID_ACTION_PENALTY  # not rejected as invalid
    assert env.game.hands_remaining == env.game.hands_per_round - 1  # the play actually happened


def test_psychic_boss_rejects_non_five_card_plays_as_invalid_action():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.blind = Blind.BOSS
    env.game.active_boss = BossBlind("The Psychic", required_play_size=5)

    obs, reward, terminated, truncated, info = env.step(_round_action((0,), mode=0))  # 1 card
    assert reward == INVALID_ACTION_PENALTY
    assert env.game.hands_remaining == env.game.hands_per_round  # rejected, no play happened


def test_psychic_boss_accepts_five_card_plays():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.blind = Blind.BOSS
    env.game.active_boss = BossBlind("The Psychic", required_play_size=5)

    obs, reward, terminated, truncated, info = env.step(_round_action((0, 1, 2, 3, 4), mode=0))
    assert reward != INVALID_ACTION_PENALTY
    assert env.game.hands_remaining == env.game.hands_per_round - 1


def test_random_policy_episode_completes_under_every_ante1_boss():
    # Broad regression guard: force each of the 8 implemented Ante 1 bosses
    # active from the start of an episode and run a random policy to
    # completion, the same kind of check that would have caught the earlier
    # Manacle stuck-loop bug for any of the newer bosses too.
    for boss in IMPLEMENTED_ANTE_1_BOSSES:
        env = BalatroEnv()
        env.reset(seed=0)
        env.game.blind = Blind.BOSS
        env.game.active_boss = boss
        env.game._start_round()

        for _ in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        else:
            raise AssertionError(f"episode under {boss.name} did not terminate within 300 steps")


def _best_hand_type(cards) -> HandType:
    from itertools import combinations
    return max(
        evaluate_hand(list(combo)).hand_type for k in range(3, 6) for combo in combinations(cards, k)
    )


def test_made_hand_bias_forces_full_house_on_reset():
    env = BalatroEnv(made_hand_bias={"FULL_HOUSE": 1.0})
    env.reset(seed=0)
    assert _best_hand_type(env.game.hand) >= HandType.FULL_HOUSE


def test_made_hand_bias_also_applies_to_later_rounds_this_episode():
    env = BalatroEnv(made_hand_bias={"FLUSH": 1.0})
    env.reset(seed=0)
    env.game.round_chips = env.game.requirement  # force this round beaten
    env.step(_round_action((0,), mode=0))  # play -> enters shop
    assert env.game.phase == "shop"

    leave_action = _shop_action(SHOP_SLOTS + 1)
    env.step(leave_action)  # leave shop -> new round should be biased too
    assert env.game.phase == "round"
    assert _best_hand_type(env.game.hand) >= HandType.FLUSH


def test_obs_exposes_best_achievable_hand_type_for_a_forced_full_house():
    env = BalatroEnv(made_hand_bias={"FULL_HOUSE": 1.0})
    obs, _ = env.reset(seed=0)
    assert _best_hand_type(env.game.hand) == HandType.FULL_HOUSE
    expected = HandType.FULL_HOUSE.value / MAX_ACHIEVABLE_HAND_TYPE
    assert obs[BEST_HAND_TYPE_OBS_INDEX] == expected


def test_obs_combo_membership_flags_match_the_search_functions_own_combo():
    from balatro_sim.env import _best_achievable_hand_type_and_combo

    env = BalatroEnv(made_hand_bias={"FULL_HOUSE": 1.0})
    obs, _ = env.reset(seed=0)
    hand = env.game.hand
    best_type, best_combo = _best_achievable_hand_type_and_combo(hand)
    assert best_type == HandType.FULL_HOUSE
    made_indices = {hand.index(c) for c in best_combo}
    for i in range(HAND_SLOTS):
        expected_flag = 1.0 if i in made_indices else 0.0
        assert obs[COMBO_MEMBERSHIP_OBS_START + i] == expected_flag


def test_obs_best_achievable_hand_type_is_zero_during_shop_phase():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.round_chips = env.game.requirement  # force this round beaten
    obs, _, _, _, _ = env.step(_round_action((0,), mode=0))  # play -> enters shop
    assert env.game.phase == "shop"
    assert obs[BEST_HAND_TYPE_OBS_INDEX] == 0.0
    for i in range(HAND_SLOTS):
        assert obs[COMBO_MEMBERSHIP_OBS_START + i] == 0.0


def test_obs_best_achievable_hand_type_tracks_a_weak_hand_low():
    env = BalatroEnv()
    obs, _ = env.reset(seed=0)
    best = _best_hand_type(env.game.hand)
    expected = min(1.0, best.value / MAX_ACHIEVABLE_HAND_TYPE)
    assert obs[BEST_HAND_TYPE_OBS_INDEX] == expected
