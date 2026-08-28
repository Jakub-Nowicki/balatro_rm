# balatro_rm

A from-scratch Python simulator of Balatro's core rules, built to train a reinforcement learning agent.

## Plan

1. Core data model (`Card`, `Deck`) — done
2. Poker hand evaluation — done, verified against balatrowiki.org
3. Scoring pipeline (chips x mult) — done, with a joker scoring-hook system (see step 6)
4. Round/ante loop (blinds, win/loss) — done, verified against balatrowiki.org
5. Shop + economy — done: interest, unused-hand bonus, reroll/buy mechanics, wired into the round loop as a phase (see step 6)
6. Curated joker subset (37: 32 common, 4 uncommon, 1 rare -- all unlocked-from-start jokers whose effects fit the current stateless-per-hand scoring hook; Tarot/Planet/Spectral-dependent and persistent-state jokers deferred) — done: extensible scoring-hook system, verified against balatrowiki.org. `GameState` now alternates between a `"round"` phase and a `"shop"` phase after every blind, so the agent actually earns money and buys jokers mid-run instead of them being fixed for the whole episode. Shop rolls each of its 2 slots by rarity weight (70% common / 25% uncommon / 5% rare, verified against balatrowiki.org) before picking uniformly within that rarity tier.
7. Gym-style env wrapper — done: `BalatroEnv`. Action space is `MultiDiscrete([218, 2, SHOP_ACTION_SIZE])` -- dim 0 indexes directly into the 218 valid 1-5-card subsets of an 8-card hand (not 8 independent binary flags; see "RL findings" below for why), dim 1 is play/discard, dim 2 is the shop decision. `enable_shop=False` collapses the env to round-only play for isolating the round-phase reward signal.
8. Baseline agents — done: random + a greedy heuristic (enumerates all achievable hands each turn, and buys the cheapest affordable joker in the shop), plus a PPO agent via Stable-Baselines3 (`scripts/train_ppo.py`, flags: `--timesteps --n-envs --device --no-shop --ent-coef --net-arch --csv-path --checkpoint-path`). Every training episode is logged to CSV via `EpisodeCsvLogger`, and `scripts/plot_progress.py` turns that into a reward/ante chart.
9. All 8 Ante 1 Boss Blind effects — done: suit debuffs (The Club/Goad/Window/Head), hand-size shrink (The Manacle), free auto-discard after each play (The Hook), exact-5-card play requirement (The Psychic), and debuffing cards played earlier this same ante (The Pillar), verified against balatrowiki.org/w/Blinds_and_Antes. This project is scoped to Ante 1 only for now; Tarot/Planet/Spectral cards and vouchers remain out of scope.
10. Real-game interface (mod API / automation) for demo play

## RL findings so far

All runs below use the `--no-shop` ablation (round-only play, no jokers) to isolate the core card-selection reward signal. random and heuristic mean_reward is ~-0.6 / ~2.3 respectively in this config -- that's the floor and the target to beat.

| run | net_arch | timesteps | final mean_reward | notes |
|---|---|---|---|---|
| exp1 (broken action space) | 64,64 | 300k | -9.5 | worse than random -- see below |
| exp1b (fixed action space) | 64,64 | 1M | -0.52 | beats random, plateaus well below heuristic |
| exp2 (bigger net) | 256,256 | 2M | -0.30 | best so far, but unstable updates (see below) |

- **Jokers matter a lot**: with the shop wired up, the heuristic agent jumps from mean_reward ~2.3 (no shop) to ~15.7 and mean_ante 1.1 -> 3.1 (shop enabled), just from buying cheap jokers. Confirms the joker/shop system is scoring correctly end-to-end.
- **Action space shape mattered more than expected for PPO**: the original action space used 8 independent binary flags for card selection, so ~15% of sampled actions selected an invalid card count and got a flat penalty regardless of *which* cards were picked -- reward carrying no information about card quality. PPO trained on that design (exp1) was actively worse than random. Re-encoding the round action as a single index into the 218 valid card subsets (so every sampled action is a legal hand) fixed that -- exp1b/exp2 are both consistently better than random, unlike exp1.
- **Environment is genuinely hard even for a strong heuristic**: with no jokers, the heuristic (which does a full combinatorial best-hand search every turn) only clears Ante 1's Small Blind about 7-12% of the time across different seeds. This isn't a bug -- vanilla Balatro's early antes are meant to be a grind without joker support. PPO cleared it exactly once in 2M steps (exp2) -- rare, but proof it's reachable, not fundamentally blocked.
- **Bigger network alone isn't a clean win**: exp2 (256,256) ended up better than exp1b (64,64) but trained for 2x the steps, and its `approx_kl` (~0.16) and `clip_fraction` (~0.66) were far above healthy PPO ranges (typically ~0.01-0.03 and <0.3) throughout -- the larger network was updating unstably at the same learning_rate=3e-4 SB3 default. Next step: retry 256,256 with a lower learning rate (or add `target_kl` to auto-throttle updates) before concluding capacity is or isn't the bottleneck.
- **Next experiments worth trying, roughly in order of expected value**: (1) lower learning_rate with the bigger net, (2) much longer training at 64,64 now that the action space is fixed, since exp1b hadn't clearly plateaued, (3) re-run the full shop-enabled env now that round-phase learning is no longer actively broken.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Training

```bash
python scripts/run_baseline.py 30 [--no-shop]
python scripts/train_ppo.py --timesteps 1000000 --n-envs 8 [--no-shop] [--net-arch 256,256] [--ent-coef 0.01]
python scripts/plot_progress.py logs/ppo_training_log.csv
```
