from __future__ import annotations

import random
from itertools import combinations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_sim.blinds import BLIND_ORDER, BossBlind
from balatro_sim.cards import Card, Suit
from balatro_sim.game_state import DISCARDS_PER_ROUND, GameState
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.jokers import JOKER_NAME_TO_INDEX, JOKER_NAMES, Joker
from balatro_sim.scoring import score_hand
from balatro_sim.shop import CARD_SLOTS as SHOP_SLOTS

HAND_SLOTS = 8
MAX_ANTE = 8  # ANTE_BASE_CHIPS only covers 1-8; reaching ante 9 ends the episode as a win
CARD_FEATURES = 5  # 1 normalized rank + 4 one-hot suit
SHOP_ACTION_SIZE = SHOP_SLOTS + 2  # buy offering[i], ..., reroll, leave shop
MAX_JOKER_PRICE_NORM = 20.0  # legendary jokers top out around here
MAX_REROLL_COST_NORM = 20.0
# Highest HandType actually achievable in this simulator (no card
# enhancements, so FIVE_OF_A_KIND/FLUSH_HOUSE/FLUSH_FIVE never occur) --
# used to normalize the best-achievable-hand-type observation feature below.
MAX_ACHIEVABLE_HAND_TYPE = HandType.STRAIGHT_FLUSH.value
OBS_SIZE = HAND_SLOTS * CARD_FEATURES + 7 + 1 + HAND_SLOTS + 1 + SHOP_SLOTS * 2 + 1

# Round-phase card selection used to be encoded as a single index into a
# fixed lookup of every legal 1-5 card subset of the 8 hand slots (218
# combinations) rather than 8 independent per-slot binary flags, specifically
# to keep every *sampled* action a real, playable hand during RL exploration
# -- a free-form multi-binary encoding lets random samples pick 0 or 6+
# cards, and every one of those got the same flat invalid penalty regardless
# of which cards were picked, which is reward signal with no information in
# it. That held up fine for RL, but turned out to make behavior cloning
# nearly unlearnable: confirmed directly across three independent attempts to
# fix a policy that kept breaking up made hands (full houses/flushes) --
# adding an explicit best-achievable-hand-type observation feature, heavily
# biasing training data toward made-hand situations, and using a much bigger
# network (256x256) -- none of it moved the needle (preservation stayed in
# the 0.5-9.5% range throughout, vs. the BC teacher's own ~90-94%), and a
# held-out accuracy check showed the model matching the heuristic's exact
# chosen subset only 5-34% of the time across every subset size, while
# nailing the simple 2-way play/discard decision 96.6% of the time. A flat
# 218-way categorical has no exploitable structure relating "index 57" to
# "index 58" -- adjacent indices don't correspond to similar hands -- so it's
# a far harder BC target than 8 independent per-card decisions, each of which
# only has to learn "should *this* card (whose own rank/suit are directly
# visible in the observation) be included," from the same raw features BC
# already learns "mode" from just fine. Switched to per-slot binary flags,
# with an explicit fallback in step() below (instead of a flat invalid
# penalty) for the malformed-selection case, to keep the original
# RL-exploration guarantee: every action, even a nonsensical one, still
# resolves to some real play with a real, information-bearing reward.

INVALID_ACTION_PENALTY = -0.05
# Rebalanced from BLIND_BEATEN_BONUS=0.5/RUN_WON_BONUS=5.0: a typical winning
# episode (~12 plays across 3 blinds) earns a cumulative per-hand reward
# (score term + hand-efficiency term below) of roughly 5-7 -- almost exactly
# the same size as the OLD flat terminal bonuses (0.5*3 + 5.0 = 6.5). That
# meant simply winning contributed as much total reward as playing every
# single hand well, diluting the per-hand shaping signal (which exists
# specifically to teach "always play your best available hand" -- see
# HAND_EFFICIENCY_WEIGHT below) relative to the sparser "did you survive"
# signal. Winning should be the consequence of playing well, not a
# comparably-sized reward in its own right, so the terminal bonuses are cut
# and the per-hand efficiency weight is doubled to make play quality clearly
# dominate the episode's total return.
BLIND_BEATEN_BONUS = 0.3
RUN_WON_BONUS = 2.0
LOSS_PENALTY = -1.0
MAX_CONSECUTIVE_INVALID = 5
# A discard scores nothing either way, so with no cost attached the agent has
# no signal that discards are a scarce resource (only 3-4 per round) -- it
# was observed discarding 3-4 times in a row with no clear plan. This small
# fixed cost makes "was this discard worth it" something the policy actually
# has to weigh, rather than a free action.
DISCARD_COST = -0.02
# The total-score reward alone wasn't dense enough for the policy to
# reliably learn "always play your best available hand" -- confirmed via
# both a live-game transcript and a simulator transcript showing the same
# pattern: a full house or flush sitting right there in hand, ignored in
# favor of a much weaker play, repeatedly, within the same round. This adds
# a per-play bonus for how close the actual play's score was to the best
# score achievable from that same dealt hand (1.0 = played optimally), which
# gives that lesson a signal on every single play, not just the rare cases
# where the natural score reward happens to make the difference obvious.
# Doubled from 0.5 as part of the terminal-bonus rebalance above -- see that
# comment for the reasoning (per-hand play quality should clearly dominate
# the episode's total return, not just be comparable to the flat win bonus).
HAND_EFFICIENCY_WEIGHT = 1.0


def _best_achievable_score_total(game: GameState, hand: list[Card]) -> int:
    """The highest score.total achievable from any subset of `hand` that's
    actually a legal play right now, under the same scoring context (jokers,
    boss, deck) `game` is currently in. Used only for the hand-efficiency
    reward bonus, not real gameplay."""
    debuffed_suit = game.active_boss.debuffed_suit if game.active_boss else None
    debuffed_cards = (
        game.cards_played_this_ante
        if game.active_boss and game.active_boss.debuffs_previously_played_cards
        else None
    )
    # The Psychic allows only exactly-5-card plays -- comparing against
    # smaller combos the policy isn't even allowed to choose would make the
    # efficiency ratio unfairly pessimistic during Psychic rounds specifically.
    required_size = game.active_boss.required_play_size if game.active_boss else None
    sizes = [required_size] if required_size is not None else range(1, min(5, len(hand)) + 1)
    best = 0
    for k in sizes:
        if k > len(hand):
            continue
        for combo in combinations(hand, k):
            result = evaluate_hand(list(combo))
            score = score_hand(
                result,
                played_cards=list(combo),
                jokers=game.jokers,
                hands_remaining=game.hands_remaining,
                discards_remaining=game.discards_remaining,
                money=game.money,
                debuffed_suit=debuffed_suit,
                debuffed_cards=debuffed_cards,
                deck_size=len(game.deck),
                max_joker_slots=game.max_joker_slots,
                rng=game.rng,
            )
            if score.total > best:
                best = score.total
    return best


def _best_achievable_hand_type_and_combo(hand: list[Card]) -> tuple[HandType, list[Card]]:
    """The strongest HandType formable from any subset of `hand` (independent
    of jokers/scoring context -- pure card composition), and the specific
    cards that form it. Used for two explicit observation features: the
    scalar HandType (confirmed via repeated diagnostics, both live-game and
    simulator transcripts across several training attempts, that the policy
    reliably fails to recognize when a strong combo -- a full house or flush
    especially -- is sitting right there in a hand made of otherwise
    ordinary-looking individual cards) and, per-card, whether each specific
    card belongs to that combo. The scalar alone told the network *that* a
    good hand exists but not *which* cards realize it -- confirmed the hard
    way: switching round-phase card selection from a single subset-index
    categorical to 8 independent per-card binary flags (see the module-level
    comment above INVALID_ACTION_PENALTY) made behavior cloning collapse to
    flagging 6-8 of 8 cards almost every time, regardless of hand content --
    worse with more training epochs, not better, ruling out
    undertraining -- because each per-card flag has to be decided from the
    same shared hidden representation with no direct signal for *that*
    card's own relevance. This per-card membership feature closes that gap:
    the output structure (one flag per card) now mirrors an input structure
    the network can directly copy from, rather than needing to rederive
    "which of these 8 scattered cards share a rank/suit" purely from raw
    rank/suit features through a small shared MLP."""
    best_type = HandType.HIGH_CARD
    best_combo: list[Card] = list(hand[:1])
    for k in range(1, min(5, len(hand)) + 1):
        for combo in combinations(hand, k):
            hand_type = evaluate_hand(list(combo)).hand_type
            if hand_type.value > best_type.value:
                best_type = hand_type
                best_combo = list(combo)
    return best_type, best_combo


def _best_achievable_hand_type(hand: list[Card]) -> HandType:
    return _best_achievable_hand_type_and_combo(hand)[0]


class BalatroEnv(gym.Env):
    """A Gym-style wrapper around GameState, alternating between a round
    phase (play/discard hands) and a shop phase (buy/reroll/skip jokers)
    after every blind.

    Action: MultiDiscrete([2]*HAND_SLOTS + [2, SHOP_ACTION_SIZE]).
      - dims 0..HAND_SLOTS-1: one binary flag per hand slot -- 1 = include
        this card in the play/discard. Read only when game.phase == "round".
        A malformed selection (0 flags set, or more than 5) is corrected
        rather than penalized as invalid -- see the module-level comment
        above INVALID_ACTION_PENALTY -- so every sampled action still
        resolves to a real play.
      - dim HAND_SLOTS: mode -- 0 = play, 1 = discard. Read only when
        game.phase == "round".
      - dim HAND_SLOTS+1: shop choice -- 0/1 = buy that shop offering,
        SHOP_SLOTS = reroll, SHOP_SLOTS+1 = leave the shop. Read only when
        game.phase == "shop". All dims are always present in the action
        (fixed-shape action space); the env ignores whichever slice doesn't
        match the current phase.

    Observation: a flat float vector -- 8 hand slots x (rank, one-hot suit),
    hands_remaining, discards_remaining, round-progress fraction, normalized
    money, normalized ante, normalized blind index, joker-slot fill fraction,
    the best achievable HandType from the current hand (normalized, 0 outside
    round phase -- see _best_achievable_hand_type_and_combo), 8 per-slot flags
    for whether that specific card belongs to the best achievable combo (all
    0 outside round phase -- same function), a shop-phase flag, up to
    SHOP_SLOTS offerings x (price, joker identity), and normalized reroll cost.

    enable_shop=False skips the shop phase transparently (auto-leaves with no
    reward), collapsing the env back to round-only play -- useful for
    isolating the round-phase reward signal from shop-decision complexity
    when debugging training.

    win_at_ante=1 (the default) ends the episode as a win once the Ante 1
    Boss Blind is cleared, rather than continuing indefinitely. Note:
    GameState currently only has a boss pool for Ante 1
    (IMPLEMENTED_ANTE_1_BOSSES) -- raising win_at_ante above 1 would reach
    Ante 2's Boss Blind still using Ante 1's boss pool, which is wrong.
    Multi-ante boss pools are a follow-up, not built yet.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jokers: list[Joker] | None = None,
        enable_shop: bool = True,
        win_at_ante: int = 1,
        boss_pool: list[BossBlind] | None = None,
        made_hand_bias: dict[str, float] | None = None,
    ):
        super().__init__()
        self._starting_jokers = jokers or []
        self.enable_shop = enable_shop
        self.win_at_ante = win_at_ante
        self.boss_pool = boss_pool
        # Training-only: {"FULL_HOUSE": 0.1, "FLUSH": 0.1} deals that hand type
        # (already complete, in the starting hand) with the given probability
        # each round instead of relying on how often it occurs naturally --
        # see GameState.force_made_hand_for_training for why.
        self.made_hand_bias = made_hand_bias
        self.action_space = spaces.MultiDiscrete([2] * HAND_SLOTS + [2, SHOP_ACTION_SIZE])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.game: GameState | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = random.Random(seed) if seed is not None else random.Random()
        # Red Deck (the default starting deck, unlocked from the start) gives
        # +1 discard per round over the game's base 3.
        self.game = GameState(
            jokers=list(self._starting_jokers),
            discards_per_round=DISCARDS_PER_ROUND + 1,
            boss_pool=self.boss_pool,
            rng=rng,
        )
        self._consecutive_invalid = 0
        self._maybe_bias_hand()
        return self._get_obs(), {}

    def _maybe_bias_hand(self) -> None:
        """Training-only: with the configured probability, forces this
        round's starting hand to already contain a made FULL_HOUSE or FLUSH.
        No-op when made_hand_bias isn't set (the default)."""
        if not self.made_hand_bias:
            return
        roll = self.game.rng.random()
        cumulative = 0.0
        for type_name, prob in self.made_hand_bias.items():
            cumulative += prob
            if roll < cumulative:
                self.game.force_made_hand_for_training(HandType[type_name])
                return

    @staticmethod
    def _encode_card(card: Card) -> list[float]:
        rank_norm = (card.rank.value - 2) / 12.0
        suit_onehot = [1.0 if card.suit is s else 0.0 for s in Suit]
        return [rank_norm, *suit_onehot]

    def _get_obs(self) -> np.ndarray:
        game = self.game
        feats: list[float] = []
        for i in range(HAND_SLOTS):
            if i < len(game.hand):
                feats.extend(self._encode_card(game.hand[i]))
            else:
                feats.extend([0.0] * CARD_FEATURES)

        progress = min(1.0, game.round_chips / game.requirement) if game.ante <= MAX_ANTE else 1.0
        feats.append(game.hands_remaining / game.hands_per_round)
        feats.append(game.discards_remaining / game.discards_per_round)
        feats.append(progress)
        feats.append(min(1.0, game.money / 50.0))
        feats.append(min(1.0, game.ante / MAX_ANTE))
        feats.append(BLIND_ORDER.index(game.blind) / (len(BLIND_ORDER) - 1))
        feats.append(min(1.0, len(game.jokers) / game.max_joker_slots))

        if game.phase == "round" and game.hand:
            hand = game.hand
            best_type, best_combo = _best_achievable_hand_type_and_combo(hand)
            feats.append(min(1.0, best_type.value / MAX_ACHIEVABLE_HAND_TYPE))
            best_combo_set = set(best_combo)
            for i in range(HAND_SLOTS):
                feats.append(1.0 if i < len(hand) and hand[i] in best_combo_set else 0.0)
        else:
            feats.append(0.0)
            feats.extend([0.0] * HAND_SLOTS)

        in_shop = game.phase == "shop" and game.shop is not None
        feats.append(1.0 if in_shop else 0.0)
        offerings = game.shop.offerings if in_shop else []
        for i in range(SHOP_SLOTS):
            if i < len(offerings):
                item = offerings[i]
                feats.append(min(1.0, item.price / MAX_JOKER_PRICE_NORM))
                feats.append(JOKER_NAME_TO_INDEX.get(item.name, 0) / max(1, len(JOKER_NAMES) - 1))
            else:
                feats.extend([0.0, 0.0])
        reroll_cost = game.shop.reroll_cost if in_shop else 0
        feats.append(min(1.0, reroll_cost / MAX_REROLL_COST_NORM))

        return np.array(feats, dtype=np.float32)

    def step(self, action):
        game = self.game
        blind_before = (game.ante, game.blind)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        if game.phase == "round":
            mode = int(action[HAND_SLOTS])
            # A flagged slot can reference a card that no longer exists when
            # a boss (e.g. The Manacle, -1 hand size) shrinks the hand below
            # 8 -- filtered out here rather than rejecting the whole action
            # (which produced a genuine stuck loop under a deterministic
            # policy re-selecting the same now-partially-invalid slots every
            # step, confirmed during the overnight run).
            selected = [i for i in range(HAND_SLOTS) if action[i] and i < len(game.hand)]
            if not selected:
                # Nothing flagged (or every flagged slot was out of range) --
                # fall back to the single highest-value card rather than a
                # flat invalid penalty, so a malformed selection still
                # resolves to a real play/discard with a real,
                # information-bearing reward. See the module-level comment
                # above INVALID_ACTION_PENALTY for why.
                selected = [max(range(len(game.hand)), key=lambda i: game.hand[i].rank.chip_value)]
            elif len(selected) > 5:
                selected = selected[:5]
            cards = [game.hand[i] for i in selected]

            # Discarding with 0 discards left can't be corrected by the agent
            # picking a different mode next step -- discards_remaining only
            # ever goes down. Without this fallback, a policy that leans
            # "discard" walks into a soft-lock: every remaining step in the
            # round is invalid, for the rest of the episode. Falling back to
            # play (using the same card selection) guarantees the round
            # always keeps moving forward, the same way a real UI would just
            # gray out the discard button and leave play as the only option.
            if mode == 1 and game.discards_remaining <= 0:
                mode = 0

            valid = True
            if mode == 0:
                valid = game.hands_remaining > 0
                # The Psychic requires playing exactly 5 cards -- anything
                # else is invalid, same as any other boss/hand-size constraint
                # the policy has to learn to respect.
                required = game.active_boss.required_play_size if game.active_boss else None
                if required is not None:
                    valid = valid and len(cards) == required
            else:
                valid = game.discards_remaining > 0

            if not valid:
                reward = INVALID_ACTION_PENALTY
            elif mode == 0:
                # Computed before play() mutates hands_remaining/discards_remaining/
                # deck/cards_played_this_ante -- those all feed into scoring (some
                # jokers and The Pillar's debuff depend on them), so the "best
                # possible" comparison has to use the same pre-play context the
                # real play was actually scored under, not the post-play one.
                best_total = _best_achievable_score_total(game, list(game.hand))
                score = game.play(cards)
                reward = score.total / max(1, game.requirement)
                if best_total > 0:
                    reward += HAND_EFFICIENCY_WEIGHT * (score.total / best_total)
                if game.is_blind_beaten:
                    game.collect_reward_and_advance()  # enters "shop" phase
                    reward += BLIND_BEATEN_BONUS
                    if game.ante > self.win_at_ante:
                        reward += RUN_WON_BONUS
                        terminated, truncated = True, True
                        info["result"] = "won_run"
                    elif not self.enable_shop:
                        game.leave_shop()  # skip within the same step, no extra step needed
                elif game.is_game_over_loss:
                    reward += LOSS_PENALTY
                    terminated = True
                    info["result"] = "lost"
            else:
                game.discard(cards)
                reward = DISCARD_COST
        elif not self.enable_shop:
            game.leave_shop()  # skip straight through, transparent to the agent
        else:  # phase == "shop"
            shop_choice = int(action[HAND_SLOTS + 1])
            try:
                offerings = game.shop.offerings
                if shop_choice < SHOP_SLOTS:
                    if shop_choice >= len(offerings):
                        raise ValueError("no offering at that slot")
                    game.buy_joker(offerings[shop_choice])
                elif shop_choice == SHOP_SLOTS:
                    game.reroll_shop()
                else:
                    game.leave_shop()
            except ValueError:
                reward = INVALID_ACTION_PENALTY

        # A repeated invalid action in an unchanging state (e.g. trying to buy
        # something unaffordable) can't self-correct under deterministic
        # evaluation the way it might by chance under stochastic training-time
        # sampling -- confirmed this produces an indefinite stuck loop for the
        # rest of the episode (a policy whose shop-decision head was never
        # meaningfully trained, e.g. one warm-started from a no-shop run,
        # walked straight into this). Force a safe exit after a few repeats,
        # the same guarantee the discard-fallback above gives the round phase.
        if reward == INVALID_ACTION_PENALTY:
            self._consecutive_invalid += 1
        else:
            self._consecutive_invalid = 0
        if self._consecutive_invalid >= MAX_CONSECUTIVE_INVALID and game.phase == "shop":
            game.leave_shop()
            self._consecutive_invalid = 0

        if game.phase == "round" and (game.ante, game.blind) != blind_before:
            self._maybe_bias_hand()

        if terminated or truncated:
            info["ante"] = game.ante

        return self._get_obs(), reward, terminated, truncated, info
