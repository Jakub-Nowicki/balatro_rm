# balatro_rm

A from-scratch Python simulator of Balatro's core rules, built to train a reinforcement learning agent.

## Plan

1. Core data model (`Card`, `Deck`) — done
2. Poker hand evaluation — done, verified against balatrowiki.org
3. Scoring pipeline (chips x mult) — done for base hands; joker hooks not yet added
4. Round/ante loop (blinds, win/loss) — done, verified against balatrowiki.org
5. Shop + economy — done: interest, unused-hand bonus, reroll/buy mechanics. Item pool is still empty pending real joker data (step 6).
6. Curated joker subset (~22) — done: extensible scoring-hook system, verified against balatrowiki.org
7. Gym-style env wrapper — done: `BalatroEnv`, flat-vector observation, MultiDiscrete action (select cards + play/discard)
8. Baseline agents — done: random + a greedy heuristic (enumerates all achievable hands each turn), plus a PPO agent via Stable-Baselines3 (`scripts/train_ppo.py`)
9. Expand fidelity (more jokers, tarot/planet/spectral, vouchers, boss blind effects, shop wired into the env)
10. Real-game interface (mod API / automation) for demo play

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pytest
```
