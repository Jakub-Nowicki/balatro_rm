# balatro_rm

A from-scratch Python simulator of Balatro (the roguelike poker deckbuilder), built to train a
PPO reinforcement learning agent — and then deploy that agent against the **real game**, live,
via a mod bridge.

## Results

The trained agent beats Ante 1 of real Balatro, playing through the actual game client (not just
the simulator) — **73.5% win rate over 200 live games** (147/200). The model itself was trained
on round-play only (no shop/jokers in its own decision-making); during live play, jokers still
get picked up along the way through a simple fixed "buy the cheapest affordable joker" rule
running alongside it, so the agent benefits from jokers in practice without ever having had to
learn shop strategy itself.

For reference, a random policy wins 0% and a greedy best-immediate-score heuristic wins under
10% under the same conditions.

## Project scope

Ante 1 only, for now. In scope: all 8 Ante 1 Boss Blind effects and the full poker hand-scoring
pipeline. Out of scope (for now): multi-ante play, the skip-blind/Tags mechanic, card
enhancements/editions.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # .venv\Scripts\activate on Windows cmd
pip install -r requirements.txt
pip install -e .
pytest
```

## Training

The main pipeline is `scripts/train_warmstart.py`: it first collects a behavior-cloning dataset
from a hand-coded heuristic, imitation-trains a policy on it (fast warm start), then continues
with real PPO fine-tuning on the actual reward signal.

```bash
# Round-play only -- this is the config the 73.5%-win-rate model was trained with
python scripts/train_warmstart.py --no-shop --made-hand-bias FULL_HOUSE,FLUSH \
    --checkpoint-path checkpoints/my_run

# Resume a stopped/crashed run from its last checkpoint
python scripts/train_warmstart.py --no-shop --init-checkpoint checkpoints/my_run_step_XXXXXX_steps.zip \
    --timesteps <steps still remaining> --checkpoint-path checkpoints/my_run
```

Key flags: `--timesteps` (default 15M), `--n-envs` (default 24), `--bc-episodes`/`--bc-epochs`
for the imitation phase, `--made-hand-bias TYPE1,TYPE2` to oversample rare strong-hand deals
during training (a random 8-card deal rarely already contains a full house/flush, so without
this the policy gets very little exposure to "you're already holding a great hand" situations),
`--net-arch 64,64`, `--device auto|cpu|cuda`. Saves an intermediate checkpoint every
`--checkpoint-every` steps (default 200k) so a crash never loses more than that.

```bash
python scripts/plot_progress.py logs/my_run_log.csv          # reward/ante chart from training
```

## Playing live, against the real game

`scripts/play_live.py` drives an actual running copy of Balatro via
[BalatroBot](https://github.com/coder/balatrobot), a mod exposing a JSON-RPC API — the same
network your trained model plays against was never trained on the real client, so this is a true
out-of-distribution test.

### One-time install

1. Install [Balatro](https://store.steampowered.com/app/2379780/Balatro/) via Steam.
2. Install the [Lovely Injector](https://github.com/ethangreen-dev/lovely-injector) (lets mods
   patch the game).
3. Install [Steamodded](https://github.com/Steamopollys/Steamodded) (the mod loader Balatro mods
   run on).
4. Install [BalatroBot](https://github.com/coder/balatrobot) into your Balatro mods folder
   (`%AppData%/Balatro/mods/` on Windows). It ships its own Python CLI/package — from inside that
   mod folder:
   ```bash
   uv sync   # or: pip install -e .
   ```

### Running it

```bash
# Terminal 1 -- from the balatrobot mod folder: launches Balatro itself with the mod loaded,
# and starts the JSON-RPC server on 127.0.0.1:12346
cd path/to/mods/balatrobot
uv run balatrobot serve --fast            # --fast = 10x game speed
# add --headless to skip rendering entirely (faster, no visible window)

# Terminal 2 -- from this repo, once you see "Balatro running on port 12346":
python scripts/play_live.py --checkpoint checkpoints/stable/<your-model>.zip \
    --num-games 20 --max-ante 1
```

Each hand it's dealt, the decision it makes, and every shop purchase get printed live, e.g.:

```
[1] hand: 0:AS 1:5H 2:4H 3:3S 4:2D  (best available: STRAIGHT)
[1] SELECTING_HAND -> play [0, 2, 4, 5, 7] (AS 5H 4H 3S 2D)
```

The shop itself is handled by a simple fixed rule, not the model: buy the cheapest affordable
joker if a slot is free, otherwise move on. The model only ever makes the card play/discard
calls — it was trained purely on round-play and never had to learn shop strategy, so this keeps
it out of decisions it has no basis for.

Note: the BalatroBot game process can occasionally crash under long rapid automated batches
(an engine-level stability issue, not something in this repo's code) — `play_live.py` retries
transient RPC failures automatically and reports a clean per-batch summary either way, so just
restart the server (same two commands above) and re-run for the remaining games if a batch cuts
short.

## How the agent sees and acts on the game

**Observation**: a flat vector — each of the 8 hand slots' (rank, suit), hands/discards
remaining, round progress, money, ante, blind index, the best poker hand type achievable from the
current hand, and *which specific cards* form that best hand (one flag per slot).

**Action**: one binary include/exclude flag per hand slot (which cards to play/discard) plus a
play-vs-discard mode bit. A malformed selection (0 or 6+ cards flagged) is auto-corrected to a
valid play rather than penalized — every action always resolves to a real move.

**Reward**: `(hand's score ÷ blind's required score)` + `(actual score ÷ best possible score from
that same hand)` on every play — the second term is what teaches "always play your best
available hand," checked against an exhaustive search of every combo the hand could form — plus
small milestone bonuses for beating a blind / winning the run, a small discard cost, and a loss
penalty.

## The made-hand-preservation fix

Early training runs reliably broke up a full house or flush sitting right in the dealt hand,
playing something much weaker instead — confirmed both in the simulator and against the real
game. Root cause turned out to be the action space, not the reward: representing "which cards to
play" as a single index into a lookup of all 218 possible card subsets gave the network an
arbitrary classification target with no structure to generalize from. Switching to independent
per-card flags didn't fix it either on its own (it collapsed to flagging almost every card). What
worked was adding the "which specific cards form the best available hand" observation feature
above, so the action structure (one flag per card) has a matching, directly copyable input signal
instead of requiring the network to rediscover card relationships from raw rank/suit features
alone. Verified: full-house/flush preservation went from under 10% correct to ~85-95% in testing,
and held up through full-scale training and 200 real games against the live client.
