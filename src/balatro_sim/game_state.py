from __future__ import annotations

import random
from dataclasses import dataclass

from balatro_sim.blinds import (
    BLIND_ORDER,
    IMPLEMENTED_ANTE_1_BOSSES,
    Blind,
    BossBlind,
    blind_requirement,
    blind_reward as get_blind_reward,
)
from balatro_sim.cards import Card, Deck
from balatro_sim.economy import UNUSED_HAND_BONUS, calculate_interest
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.jokers import Joker, make_joker, shop_item_pool
from balatro_sim.scoring import ScoreResult, score_hand
from balatro_sim.shop import Shop, ShopItem

@dataclass
class CashOutResult:
    blind_reward: int
    hand_bonus: int
    interest: int
    total: int


STARTING_MONEY = 4
HAND_SIZE = 8
HANDS_PER_ROUND = 4
DISCARDS_PER_ROUND = 3  # Red Deck (+1 discard) is applied by callers, not hardcoded here
MAX_JOKER_SLOTS = 5


class GameState:
    def __init__(
        self,
        ante: int = 1,
        blind: Blind = Blind.SMALL,
        hand_size: int = HAND_SIZE,
        hands_per_round: int = HANDS_PER_ROUND,
        discards_per_round: int = DISCARDS_PER_ROUND,
        starting_money: int = STARTING_MONEY,
        jokers: list[Joker] | None = None,
        max_joker_slots: int = MAX_JOKER_SLOTS,
        shop_rarities: set[str] | None = frozenset({"common", "uncommon"}),
        boss_pool: list[BossBlind] | None = None,
        rng: random.Random | None = None,
    ):
        self.ante = ante
        self.blind = blind
        self.hand_size = hand_size
        self.hands_per_round = hands_per_round
        self.discards_per_round = discards_per_round
        self.money = starting_money
        self.jokers = jokers if jokers is not None else []
        self.max_joker_slots = max_joker_slots
        self.shop_rarities = shop_rarities
        # Which bosses can be drawn for the Ante 1 Boss Blind, and at what
        # relative frequency (repeat an entry to weight it higher) -- defaults
        # to one of each, uniformly. Lets a fine-tuning run bias exposure
        # toward specific bosses without excluding the others entirely (full
        # exclusion risks the policy forgetting bosses it no longer sees).
        self.boss_pool = boss_pool if boss_pool is not None else IMPLEMENTED_ANTE_1_BOSSES
        self.rng = rng or random.Random()
        self.phase = "round"
        self.shop: Shop | None = None
        self.active_boss: BossBlind | None = None
        self.cards_played_this_ante: set[Card] = set()
        self._start_round()

    def _start_round(self) -> None:
        self.deck = Deck()
        self.deck.shuffle(self.rng)
        self.hand: list[Card] = []
        self.hands_remaining = self.hands_per_round
        self.discards_remaining = self.discards_per_round
        self.round_chips = 0
        self._draw_up_to_hand_size()

    def _effective_hand_size(self) -> int:
        delta = self.active_boss.hand_size_delta if self.active_boss else 0
        return max(1, self.hand_size + delta)

    def force_made_hand_for_training(self, hand_type: HandType) -> None:
        """Training-only utility, not part of normal game rules: replaces the
        current hand with one that already contains a complete made hand of
        the given type (FULL_HOUSE or FLUSH), mixed with random filler.

        A random 8-card deal rarely happens to already contain a complete
        made hand, so a policy gets very little natural exposure to "you
        already have a flush, don't discard into it" -- this lets a
        fine-tuning run deal that exact situation far more often than it
        occurs naturally, without changing anything about how a normal round
        is dealt. Never called during ordinary play.
        """
        pool = list(self.deck.cards) + list(self.hand)
        self.rng.shuffle(pool)

        made: list[Card] = []
        if hand_type == HandType.FULL_HOUSE:
            by_rank: dict = {}
            for c in pool:
                by_rank.setdefault(c.rank, []).append(c)
            trip_ranks = [r for r, cs in by_rank.items() if len(cs) >= 3]
            trip_rank = self.rng.choice(trip_ranks)
            trips = by_rank[trip_rank][:3]
            pair_ranks = [r for r, cs in by_rank.items() if r != trip_rank and len(cs) >= 2]
            pair_rank = self.rng.choice(pair_ranks)
            made = trips + by_rank[pair_rank][:2]
        elif hand_type == HandType.FLUSH:
            by_suit: dict = {}
            for c in pool:
                by_suit.setdefault(c.suit, []).append(c)
            for _ in range(20):
                suit = self.rng.choice([s for s, cs in by_suit.items() if len(cs) >= 5])
                candidate = by_suit[suit][:5]
                if evaluate_hand(candidate).hand_type == HandType.FLUSH:
                    made = candidate
                    break
            if not made:
                return  # couldn't find a clean (non-straight) flush this deck -- leave hand as-is
        else:
            raise ValueError(f"unsupported hand_type for force_made_hand_for_training: {hand_type}")

        remaining = [c for c in pool if c not in made]
        self.rng.shuffle(remaining)
        filler = remaining[: self._effective_hand_size() - len(made)]
        new_hand = made + filler
        self.rng.shuffle(new_hand)  # don't let the made hand always land in the same slots
        self.hand = new_hand
        self.deck.cards = [c for c in pool if c not in new_hand]

    def _draw_up_to_hand_size(self) -> None:
        need = self._effective_hand_size() - len(self.hand)
        if need > 0:
            self.hand.extend(self.deck.draw(need))

    @property
    def requirement(self) -> int:
        return blind_requirement(self.ante, self.blind)

    @property
    def is_blind_beaten(self) -> bool:
        return self.round_chips >= self.requirement

    @property
    def is_round_over(self) -> bool:
        return self.is_blind_beaten or self.hands_remaining == 0

    @property
    def is_game_over_loss(self) -> bool:
        return self.hands_remaining == 0 and not self.is_blind_beaten

    def play(self, cards: list[Card]) -> ScoreResult:
        if self.phase != "round":
            raise ValueError("not in round phase")
        if self.hands_remaining <= 0:
            raise ValueError("no hands remaining this round")
        if not cards or len(cards) > 5:
            raise ValueError("must play between 1 and 5 cards")
        if self.active_boss and self.active_boss.required_play_size is not None:
            if len(cards) != self.active_boss.required_play_size:
                raise ValueError(f"must play exactly {self.active_boss.required_play_size} cards")
        for c in cards:
            if c not in self.hand:
                raise ValueError(f"{c!r} is not in hand")

        result = evaluate_hand(cards)
        debuffed_cards = (
            self.cards_played_this_ante
            if self.active_boss and self.active_boss.debuffs_previously_played_cards
            else None
        )
        score = score_hand(
            result,
            played_cards=cards,
            jokers=self.jokers,
            hands_remaining=self.hands_remaining,
            discards_remaining=self.discards_remaining,
            money=self.money,
            debuffed_suit=self.active_boss.debuffed_suit if self.active_boss else None,
            debuffed_cards=debuffed_cards,
            deck_size=len(self.deck),
            max_joker_slots=self.max_joker_slots,
            rng=self.rng,
        )
        self.round_chips += score.total
        self.cards_played_this_ante.update(cards)

        for c in cards:
            self.hand.remove(c)
        self.hands_remaining -= 1
        self._draw_up_to_hand_size()

        # The Hook: 2 random cards are auto-discarded (free, doesn't touch
        # discards_remaining) after every played hand, as long as the round
        # isn't already over -- no point discarding into a round that just ended.
        if self.active_boss and self.active_boss.auto_discard_after_play and not self.is_round_over:
            n = min(self.active_boss.auto_discard_after_play, len(self.hand))
            for c in self.rng.sample(self.hand, n):
                self.hand.remove(c)
            self._draw_up_to_hand_size()

        return score

    def discard(self, cards: list[Card]) -> None:
        if self.phase != "round":
            raise ValueError("not in round phase")
        if self.discards_remaining <= 0:
            raise ValueError("no discards remaining this round")
        if not cards or len(cards) > 5:
            raise ValueError("must discard between 1 and 5 cards")
        for c in cards:
            if c not in self.hand:
                raise ValueError(f"{c!r} is not in hand")

        for c in cards:
            self.hand.remove(c)
        self.discards_remaining -= 1
        self._draw_up_to_hand_size()

    def collect_reward_and_advance(self) -> CashOutResult:
        if not self.is_blind_beaten:
            raise ValueError("cannot advance: blind not beaten")

        blind_reward = get_blind_reward(self.blind)
        hand_bonus = self.hands_remaining * UNUSED_HAND_BONUS
        interest = calculate_interest(self.money)
        total = blind_reward + hand_bonus + interest
        self.money += total

        idx = BLIND_ORDER.index(self.blind)
        if idx + 1 < len(BLIND_ORDER):
            self.blind = BLIND_ORDER[idx + 1]
        else:
            self.blind = Blind.SMALL
            self.ante += 1
            self.cards_played_this_ante = set()

        self.active_boss = self.rng.choice(self.boss_pool) if self.blind == Blind.BOSS else None

        self.phase = "shop"
        self.shop = Shop(shop_item_pool(self.shop_rarities), rng=self.rng)
        return CashOutResult(blind_reward=blind_reward, hand_bonus=hand_bonus, interest=interest, total=total)

    def buy_joker(self, item: ShopItem) -> None:
        if self.phase != "shop" or self.shop is None:
            raise ValueError("not in shop phase")
        if len(self.jokers) >= self.max_joker_slots:
            raise ValueError("no joker slots available")
        price = self.shop.buy(item, self.money)
        self.money -= price
        self.jokers.append(make_joker(item.name))

    def reroll_shop(self) -> None:
        if self.phase != "shop" or self.shop is None:
            raise ValueError("not in shop phase")
        cost = self.shop.reroll(self.money)
        self.money -= cost

    def leave_shop(self) -> None:
        if self.phase != "shop":
            raise ValueError("not in shop phase")
        self.phase = "round"
        self.shop = None
        self._start_round()
