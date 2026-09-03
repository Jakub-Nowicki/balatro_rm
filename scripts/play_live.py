"""Drives a real, running Balatro game via the BalatroBot JSON-RPC API
(https://github.com/coder/balatrobot), using our trained PPO model for
card play/discard decisions.

Requires BalatroBot's server already running (`uv run balatrobot serve` from
its mod folder) with Balatro open and connected.

By default the shop phase is handled by a simple heuristic, not the trained
model: it buys the cheapest affordable joker if a slot is free, otherwise
moves on. Pass --model-shop to let a checkpoint trained with the shop
enabled make its own buy, reroll, and leave decisions instead. See
choose_shop_action() below.

Usage: python scripts/play_live.py [--checkpoint PATH] [--max-ante N] [--rpc-url URL] [--model-shop]
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import requests
from stable_baselines3 import PPO

from itertools import combinations

from balatro_sim.blinds import BLIND_ORDER, Blind
from balatro_sim.cards import Card, Rank, Suit
from balatro_sim.env import CARD_FEATURES, HAND_SLOTS, MAX_ACHIEVABLE_HAND_TYPE, MAX_ANTE, SHOP_SLOTS
from balatro_sim.game_state import DISCARDS_PER_ROUND, HANDS_PER_ROUND, MAX_JOKER_SLOTS
from balatro_sim.hands import HandType, evaluate_hand
from balatro_sim.jokers import JOKER_NAME_TO_INDEX, JOKER_NAMES

DEFAULT_RPC_URL = "http://127.0.0.1:12346"
MAX_JOKER_PRICE_NORM = 20.0
MAX_REROLL_COST_NORM = 20.0
MAX_TOTAL_ACTIONS = 3000  # safety cap against an unexpected stuck loop

RANK_MAP = {
    "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
    "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
    "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
}
BLIND_KEY_TO_ENUM = {"small": Blind.SMALL, "big": Blind.BIG, "boss": Blind.BOSS}


class BalatroBotClient:
    def __init__(self, url: str, timeout: float = 60.0, max_retries: int = 3):
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries

    def call(self, method: str, params: dict | None = None) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url,
                    json={"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1},
                    timeout=self.timeout,
                )
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"RPC '{method}' failed: {data['error']}")
                return data["result"]
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                wait = 2**attempt
                print(f"  [rpc] '{method}' timed out/failed (attempt {attempt + 1}/{self.max_retries}), "
                      f"retrying in {wait}s: {exc}", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"RPC '{method}' failed after {self.max_retries} attempts") from last_exc


def encode_card(card: dict) -> list[float]:
    rank = RANK_MAP[card["value"]["rank"]]
    suit_code = card["value"]["suit"]  # already "H"/"D"/"C"/"S", matches Suit enum's .value
    rank_norm = (rank.value - 2) / 12.0
    suit_onehot = [1.0 if suit_code == s.value else 0.0 for s in Suit]
    return [rank_norm, *suit_onehot]


def to_our_card(card: dict) -> Card:
    rank = RANK_MAP[card["value"]["rank"]]
    suit = next(s for s in Suit if s.value == card["value"]["suit"])
    return Card(rank, suit)


def best_available_hand_type_and_combo(hand_cards: list[dict]) -> tuple[HandType, list[Card]]:
    """The best poker hand type actually achievable from the current hand,
    independent of what the model chose, and the specific cards that form
    it. Used both for the printed diagnostic line and as two of the
    observation features in build_obs() below."""
    cards = [to_our_card(c) for c in hand_cards]
    best_type = HandType.HIGH_CARD
    best_combo: list[Card] = cards[:1]
    for k in range(1, min(5, len(cards)) + 1):
        for combo in combinations(cards, k):
            hand_type = evaluate_hand(list(combo)).hand_type
            if hand_type.value > best_type.value:
                best_type = hand_type
                best_combo = list(combo)
    return best_type, best_combo


def best_available_hand_type(hand_cards: list[dict]) -> HandType:
    return best_available_hand_type_and_combo(hand_cards)[0]


def hand_str(hand_cards: list[dict]) -> str:
    """Renders the hand as e.g. '0:AS 1:KH 2:2D ...' for direct comparison against the on-screen hand."""
    parts = []
    for i, c in enumerate(hand_cards):
        v = c["value"]
        parts.append(f"{i}:{v['rank']}{v['suit']}")
    return " ".join(parts)


def current_blind_key(state: dict) -> str:
    for key in ("small", "big", "boss"):
        if state["blinds"][key]["status"] in ("CURRENT", "SELECT"):
            return key
    raise RuntimeError(f"no current/select blind found: {state['blinds']}")


def build_obs(state: dict) -> np.ndarray:
    """Mirrors BalatroEnv._get_obs()'s exact feature order and normalization,
    since the model was trained on that encoding and expects it exactly."""
    feats: list[float] = []
    hand_cards = state.get("hand", {}).get("cards", [])
    for i in range(HAND_SLOTS):
        feats.extend(encode_card(hand_cards[i]) if i < len(hand_cards) else [0.0] * CARD_FEATURES)

    round_info = state.get("round", {})
    hands_left = round_info.get("hands_left", 0)
    discards_left = round_info.get("discards_left", 0)
    chips = round_info.get("chips", 0)

    blind_key = current_blind_key(state)
    requirement = state["blinds"][blind_key]["score"]
    progress = min(1.0, chips / requirement) if requirement else 0.0

    feats.append(hands_left / HANDS_PER_ROUND)
    feats.append(discards_left / (DISCARDS_PER_ROUND + 1))  # Red Deck: +1 discard
    feats.append(progress)
    feats.append(min(1.0, state["money"] / 50.0))
    feats.append(min(1.0, state["ante_num"] / MAX_ANTE))
    feats.append(BLIND_ORDER.index(BLIND_KEY_TO_ENUM[blind_key]) / (len(BLIND_ORDER) - 1))

    jokers_area = state.get("jokers") or {"cards": [], "limit": MAX_JOKER_SLOTS}
    feats.append(min(1.0, len(jokers_area.get("cards", [])) / max(1, jokers_area.get("limit", MAX_JOKER_SLOTS))))

    if state["state"] == "SELECTING_HAND" and hand_cards:
        best_type, best_combo = best_available_hand_type_and_combo(hand_cards)
        feats.append(min(1.0, best_type.value / MAX_ACHIEVABLE_HAND_TYPE))
        best_combo_set = set(best_combo)  # Card is a frozen (hashable, value-equal) dataclass
        for i in range(HAND_SLOTS):
            in_combo = i < len(hand_cards) and to_our_card(hand_cards[i]) in best_combo_set
            feats.append(1.0 if in_combo else 0.0)
    else:
        feats.append(0.0)
        feats.extend([0.0] * HAND_SLOTS)

    in_shop = state["state"] == "SHOP"
    feats.append(1.0 if in_shop else 0.0)
    shop_cards = (state.get("shop") or {}).get("cards", []) if in_shop else []
    for i in range(SHOP_SLOTS):
        if i < len(shop_cards):
            item = shop_cards[i]
            feats.append(min(1.0, item["cost"]["buy"] / MAX_JOKER_PRICE_NORM))
            feats.append(JOKER_NAME_TO_INDEX.get(item.get("label", ""), 0) / max(1, len(JOKER_NAMES) - 1))
        else:
            feats.extend([0.0, 0.0])
    reroll_cost = round_info.get("reroll_cost", 0) if in_shop else 0
    feats.append(min(1.0, reroll_cost / MAX_REROLL_COST_NORM))

    return np.array(feats, dtype=np.float32)


def choose_round_action(model, state: dict) -> tuple[list[int], int]:
    obs = build_obs(state)
    action, _ = model.predict(obs, deterministic=True)
    mode = int(action[HAND_SLOTS])
    hand_size = len(state["hand"]["cards"])
    indices = [i for i in range(HAND_SLOTS) if action[i] and i < hand_size]
    if not indices:
        indices = [0]  # fall back to the first card, matching env.py's fallback
    elif len(indices) > 5:
        indices = indices[:5]
    if mode == 1 and state["round"].get("discards_left", 0) <= 0:
        mode = 0  # no discards left, fall back to playing
    return indices, mode


def choose_shop_purchase(state: dict) -> int | None:
    """Cheapest affordable joker heuristic, not the trained model (see module
    docstring). The shop can also offer Tarot/Planet/Spectral cards, which
    are filtered out here since the model has no use for them."""
    jokers_area = state.get("jokers") or {"cards": [], "limit": MAX_JOKER_SLOTS}
    if len(jokers_area.get("cards", [])) >= jokers_area.get("limit", MAX_JOKER_SLOTS):
        return None
    shop_cards = (state.get("shop") or {}).get("cards", [])
    money = state["money"]
    affordable = [
        (i, c) for i, c in enumerate(shop_cards) if c.get("set") == "JOKER" and c["cost"]["buy"] <= money
    ]
    if not affordable:
        return None
    idx, _ = min(affordable, key=lambda pair: pair[1]["cost"]["buy"])
    return idx


def choose_shop_action(model, state: dict) -> int:
    """The trained model's own shop decision: 0 to SHOP_SLOTS-1 buys that
    offering, SHOP_SLOTS rerolls, SHOP_SLOTS+1 leaves. Only meaningful for a
    checkpoint that was trained with the shop enabled, since a no-shop
    checkpoint never saw a real shop observation during training."""
    obs = build_obs(state)
    action, _ = model.predict(obs, deterministic=True)
    return int(action[HAND_SLOTS + 1])


def _unique_run_seed() -> str:
    # Balatro's own auto-generated seed collides under rapid automated calls,
    # so generate an explicit one from the OS entropy source instead.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.SystemRandom().choices(alphabet, k=8))


def play_game(client: BalatroBotClient, model, max_ante: int, game_num: int = 1, model_shop: bool = False) -> dict:
    state = client.call("start", {"deck": "RED", "stake": "WHITE", "seed": _unique_run_seed()})
    print(f"=== Game {game_num}: started run. seed={state.get('seed')} ===", flush=True)

    for step in range(MAX_TOTAL_ACTIONS):
        phase = state["state"]

        if phase == "BLIND_SELECT":
            blind_key = current_blind_key(state)
            print(f"[{step}] BLIND_SELECT -> selecting {blind_key}", flush=True)
            state = client.call("select")

        elif phase == "SELECTING_HAND":
            best_type = best_available_hand_type(state["hand"]["cards"])
            print(f"[{step}] hand: {hand_str(state['hand']['cards'])}  (best available: {best_type.name})", flush=True)
            indices, mode = choose_round_action(model, state)
            action_name = "discard" if mode == 1 else "play"
            chosen = [state["hand"]["cards"][i] for i in indices]
            chosen_str = " ".join(f"{c['value']['rank']}{c['value']['suit']}" for c in chosen)
            print(f"[{step}] SELECTING_HAND -> {action_name} {indices} ({chosen_str})", flush=True)
            state = client.call(action_name, {"cards": indices})

        elif phase == "ROUND_EVAL":
            print(f"[{step}] ROUND_EVAL -> cash_out", flush=True)
            state = client.call("cash_out")
            if state.get("ante_num", 1) > max_ante:
                print(f"=== Game {game_num}: cleared Ante {max_ante}! Stopping here. ===\n", flush=True)
                result = {"won": True, "ante_num": state.get("ante_num"), "round_num": state.get("round_num")}
                client.call("menu")
                return result

        elif phase == "SHOP" and model_shop:
            # The model can buy or reroll more than once per shop visit,
            # same as during training. Capped so it can't loop forever.
            for _ in range(6):
                shop_choice = choose_shop_action(model, state)
                shop_cards = (state.get("shop") or {}).get("cards", [])
                money = state["money"]
                jokers_area = state.get("jokers") or {"cards": [], "limit": MAX_JOKER_SLOTS}
                slots_full = len(jokers_area.get("cards", [])) >= jokers_area.get("limit", MAX_JOKER_SLOTS)
                reroll_cost = state.get("round", {}).get("reroll_cost", 0)

                if shop_choice < SHOP_SLOTS and shop_choice < len(shop_cards) \
                        and shop_cards[shop_choice]["cost"]["buy"] <= money and not slots_full:
                    item = shop_cards[shop_choice]
                    print(f"[{step}] SHOP -> buying '{item.get('label')}' (${item['cost']['buy']}) [model]", flush=True)
                    state = client.call("buy", {"card": shop_choice})
                elif shop_choice == SHOP_SLOTS and reroll_cost <= money:
                    print(f"[{step}] SHOP -> rerolling [model]", flush=True)
                    state = client.call("reroll")
                else:
                    print(f"[{step}] SHOP -> leaving [model]", flush=True)
                    break
                time.sleep(0.05)  # same settle delay as the outer loop, needed between shop calls too
                if state["state"] != "SHOP":
                    break
            state = client.call("next_round")

        elif phase == "SHOP":
            buy_idx = choose_shop_purchase(state)
            if buy_idx is not None:
                item = state["shop"]["cards"][buy_idx]
                print(f"[{step}] SHOP -> buying '{item.get('label')}' (${item['cost']['buy']})", flush=True)
                state = client.call("buy", {"card": buy_idx})
            else:
                print(f"[{step}] SHOP -> nothing affordable/free slot, moving on", flush=True)
            state = client.call("next_round")

        else:
            state = client.call("gamestate")

        if state["state"] == "GAME_OVER":
            break

        time.sleep(0.05)  # let the game's own animations/transitions settle

    won = state.get("won", False)
    print(f"=== Game {game_num}: GAME_OVER. won={won} ante={state.get('ante_num')} round={state.get('round_num')} ===\n", flush=True)
    client.call("menu")  # GAME_OVER doesn't auto-return to MENU; next start() requires it
    return {"won": won, "ante_num": state.get("ante_num"), "round_num": state.get("round_num")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/exp9_full_ante1.zip")
    parser.add_argument("--max-ante", type=int, default=1)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--model-shop", action="store_true",
                         help="let the model make its own shop buy/reroll/leave decisions "
                              "(only meaningful for a checkpoint trained with the shop enabled)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"loading {args.checkpoint}", flush=True)
    model = PPO.load(args.checkpoint, device="cpu")
    client = BalatroBotClient(args.rpc_url)
    client.call("health")
    print("connected to BalatroBot", flush=True)

    current = client.call("gamestate")
    if current["state"] != "MENU":
        print(f"game was in state '{current['state']}', resetting to MENU", flush=True)
        client.call("menu")

    results = []
    skipped = 0
    for i in range(1, args.num_games + 1):
        try:
            results.append(play_game(client, model, args.max_ante, game_num=i, model_shop=args.model_shop))
        except RuntimeError as exc:
            skipped += 1
            print(f"=== Game {i}: SKIPPED after repeated RPC failures: {exc} ===\n", flush=True)
            try:
                client.call("menu")  # best-effort recovery so the next game can still start
            except RuntimeError:
                print("=== could not reach the server to reset to MENU; stopping the batch here ===")
                break

    if args.num_games > 1 and results:
        wins = sum(1 for r in results if r["won"])
        antes = [r["ante_num"] for r in results if r["ante_num"] is not None]
        print(f"=== SUMMARY: {wins}/{len(results)} won ({100 * wins / len(results):.1f}%), {skipped} skipped ===")
        if antes:
            print(f"=== mean_ante: {sum(antes) / len(antes):.2f} ===")
    elif args.num_games > 1:
        print(f"=== SUMMARY: 0/0 won (no games completed), {skipped} skipped ===")


if __name__ == "__main__":
    main()
