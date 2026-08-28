import numpy as np

from balatro_sim.blinds import IMPLEMENTED_ANTE_1_BOSSES, Blind, BossBlind
from balatro_sim.env import CARD_SUBSETS, DISCARD_COST, OBS_SIZE, SHOP_SLOTS, BalatroEnv


def _round_action(subset: tuple[int, ...], mode: int) -> np.ndarray:
    subset_idx = CARD_SUBSETS.index(subset)
    return np.array([subset_idx, mode, 0], dtype=int)


def test_reset_returns_correctly_shaped_obs():
    env = BalatroEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert info == {}


def test_action_space_and_observation_space_shapes():
    env = BalatroEnv()
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (OBS_SIZE,)


def test_every_sampled_round_action_selects_a_valid_card_count():
    env = BalatroEnv()
    env.reset(seed=0)
    for _ in range(200):
        subset_idx = env.action_space.sample()[0]
        assert 1 <= len(CARD_SUBSETS[subset_idx]) <= 5


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

    shop_action = np.array([0, 0, 0], dtype=int)  # buy offering[0]
    env.step(shop_action)

    assert len(env.game.jokers) == starting_jokers + 1
    assert env.game.money == 100 - item.price


def test_shop_phase_leave_returns_to_round_phase():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.round_chips = env.game.requirement
    env.step(_round_action((0,), mode=0))
    assert env.game.phase == "shop"

    leave_action = np.array([0, 0, SHOP_SLOTS + 1], dtype=int)
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


def test_subset_referencing_a_shrunk_hand_slot_still_plays_the_cards_that_exist():
    # Regression test: under a hand-size-reducing boss (The Manacle), a
    # subset that references the now-missing slot used to be rejected as
    # invalid outright -- if a deterministic policy kept re-selecting that
    # same subset (same observation, same argmax action every step, nothing
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

    # needs a partial overlap (some valid indices alongside the missing one) --
    # a subset that's *only* {7} would correctly stay invalid, since nothing
    # would be left to play once that slot is filtered out.
    subset_including_missing_slot_7 = next(s for s in CARD_SUBSETS if 7 in s and len(s) > 1)
    action = np.array([CARD_SUBSETS.index(subset_including_missing_slot_7), 0, 0], dtype=int)
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward != -0.05  # not rejected as invalid
    assert env.game.hands_remaining == env.game.hands_per_round - 1  # the play actually happened


def test_psychic_boss_rejects_non_five_card_plays_as_invalid_action():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.blind = Blind.BOSS
    env.game.active_boss = BossBlind("The Psychic", required_play_size=5)

    obs, reward, terminated, truncated, info = env.step(_round_action((0,), mode=0))  # 1 card
    assert reward == -0.05
    assert env.game.hands_remaining == env.game.hands_per_round  # rejected, no play happened


def test_psychic_boss_accepts_five_card_plays():
    env = BalatroEnv()
    env.reset(seed=0)
    env.game.blind = Blind.BOSS
    env.game.active_boss = BossBlind("The Psychic", required_play_size=5)

    five_card_subset = next(s for s in CARD_SUBSETS if len(s) == 5)
    obs, reward, terminated, truncated, info = env.step(_round_action(five_card_subset, mode=0))
    assert reward != -0.05
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
