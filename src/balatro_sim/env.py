from __future__ import annotations

import random
from itertools import combinations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_sim.blinds import BLIND_ORDER
from balatro_sim.cards import Card, Suit
from balatro_sim.game_state import DISCARDS_PER_ROUND, GameState
from balatro_sim.jokers import JOKER_NAME_TO_INDEX, JOKER_NAMES, Joker
from balatro_sim.shop import CARD_SLOTS as SHOP_SLOTS

HAND_SLOTS = 8
MAX_ANTE = 8  # ANTE_BASE_CHIPS only covers 1-8; reaching ante 9 ends the episode as a win
CARD_FEATURES = 5  # 1 normalized rank + 4 one-hot suit
SHOP_ACTION_SIZE = SHOP_SLOTS + 2  # buy offering[i], ..., reroll, leave shop
MAX_JOKER_PRICE_NORM = 20.0  # legendary jokers top out around here
MAX_REROLL_COST_NORM = 20.0
OBS_SIZE = HAND_SLOTS * CARD_FEATURES + 7 + 1 + SHOP_SLOTS * 2 + 1

# Every legal 1-5 card subset of the 8 hand slots, as an index-based lookup so
# the round action is "pick a subset by index" rather than 8 independent
# binary flags. A free-form multi-binary encoding lets ~15% of raw samples
# select 0 or 6+ cards, and every one of those gets the *same* flat invalid
# penalty regardless of which cards were picked -- that's reward signal with
# no information in it, and it was drowning out the (highly learnable) signal
# for which cards are actually good. Indexing into this table makes every
# sampled action a real, playable hand: C(8,1)+...+C(8,5) = 218 subsets.
CARD_SUBSETS: list[tuple[int, ...]] = [
    combo for k in range(1, 6) for combo in combinations(range(HAND_SLOTS), k)
]
N_CARD_SUBSETS = len(CARD_SUBSETS)
SUBSET_TO_INDEX: dict[frozenset[int], int] = {frozenset(s): i for i, s in enumerate(CARD_SUBSETS)}

INVALID_ACTION_PENALTY = -0.05
BLIND_BEATEN_BONUS = 0.5
RUN_WON_BONUS = 5.0
LOSS_PENALTY = -1.0
MAX_CONSECUTIVE_INVALID = 5
# A discard scores nothing either way, so with no cost attached the agent has
# no signal that discards are a scarce resource (only 3-4 per round) -- it
# was observed discarding 3-4 times in a row with no clear plan. This small
# fixed cost makes "was this discard worth it" something the policy actually
# has to weigh, rather than a free action.
DISCARD_COST = -0.02


class BalatroEnv(gym.Env):
    """A Gym-style wrapper around GameState, alternating between a round
    phase (play/discard hands) and a shop phase (buy/reroll/skip jokers)
    after every blind.

    Action: MultiDiscrete([N_CARD_SUBSETS, 2, SHOP_ACTION_SIZE]).
      - dim 0: index into CARD_SUBSETS -- which 1-5 hand slots to act on.
        Read only when game.phase == "round".
      - dim 1: mode -- 0 = play, 1 = discard. Read only when
        game.phase == "round".
      - dim 2: shop choice -- 0/1 = buy that shop offering, SHOP_SLOTS =
        reroll, SHOP_SLOTS+1 = leave the shop. Read only when
        game.phase == "shop". All three dims are always present in the
        action (fixed-shape action space); the env ignores whichever slice
        doesn't match the current phase.

    Observation: a flat float vector -- 8 hand slots x (rank, one-hot suit),
    hands_remaining, discards_remaining, round-progress fraction, normalized
    money, normalized ante, normalized blind index, joker-slot fill fraction,
    a shop-phase flag, up to SHOP_SLOTS offerings x (price, joker identity),
    and normalized reroll cost.

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

    def __init__(self, jokers: list[Joker] | None = None, enable_shop: bool = True, win_at_ante: int = 1):
        super().__init__()
        self._starting_jokers = jokers or []
        self.enable_shop = enable_shop
        self.win_at_ante = win_at_ante
        self.action_space = spaces.MultiDiscrete([N_CARD_SUBSETS, 2, SHOP_ACTION_SIZE])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.game: GameState | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = random.Random(seed) if seed is not None else random.Random()
        # Red Deck (the default starting deck, unlocked from the start) gives
        # +1 discard per round over the game's base 3.
        self.game = GameState(
            jokers=list(self._starting_jokers), discards_per_round=DISCARDS_PER_ROUND + 1, rng=rng
        )
        self._consecutive_invalid = 0
        return self._get_obs(), {}

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
        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        if game.phase == "round":
            subset_idx, mode = int(action[0]), int(action[1])
            indices = CARD_SUBSETS[subset_idx]
            cards = [game.hand[i] for i in indices if i < len(game.hand)]

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

            # A subset can reference a slot that no longer exists when a boss
            # (e.g. The Manacle, -1 hand size) shrinks the hand below 8 --
            # `cards` above already drops those indices. Rather than reject
            # the whole action (which produced a genuine stuck loop: a
            # deterministic policy re-selecting the same now-partially-invalid
            # subset every step, capped only by hitting the 500-step episode
            # limit, confirmed during the overnight run), just play/discard
            # whichever of the selected cards still exist -- a reasonable
            # interpretation of the intent, and always possible to satisfy.
            valid = 1 <= len(cards) <= 5
            if mode == 0:
                valid = valid and game.hands_remaining > 0
                # The Psychic requires playing exactly 5 cards -- anything
                # else is invalid, same as any other boss/hand-size constraint
                # the policy has to learn to respect.
                required = game.active_boss.required_play_size if game.active_boss else None
                if required is not None:
                    valid = valid and len(cards) == required
            else:
                valid = valid and game.discards_remaining > 0

            if not valid:
                reward = INVALID_ACTION_PENALTY
            elif mode == 0:
                score = game.play(cards)
                reward = score.total / max(1, game.requirement)
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
            shop_choice = int(action[2])
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

        if terminated or truncated:
            info["ante"] = game.ante

        return self._get_obs(), reward, terminated, truncated, info
