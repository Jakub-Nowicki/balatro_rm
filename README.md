# balatro_rm

This project builds a Python simulator of Balatro (the roguelike poker card game) from scratch,
and trains an AI agent to play it using reinforcement learning. The trained agent can then play
the real game live, through a mod that connects to the actual Balatro app.

## Results

The trained agent wins Ante 1 of real Balatro about 73.5% of the time. That number comes from
200 real games played against the actual game client, not just the simulator.

The AI itself only learned how to play cards (what to keep, what to discard). It never learned
shop strategy. During live games, a simple fixed rule buys the cheapest joker it can afford
whenever one is available, and the AI still benefits from having jokers in play even though it
never had to figure out shop decisions on its own.

For comparison, a random policy wins 0% of the time, and a simple "always play the best scoring
hand" heuristic wins less than 10% of the time under the same setup.

## What is in scope

Right now this project only covers Ante 1. It includes all 8 Ante 1 Boss Blinds and the full
scoring system for poker hands. It does not yet cover later antes, the skip blind and Tags
system, or card enhancements and editions.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # on Windows cmd use: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Training the model

The main training script is `scripts/train_warmstart.py`. It works in two steps. First it
watches a simple hand coded heuristic play many games and learns to copy it (this is called
behavior cloning). Then it keeps training with real reinforcement learning, using actual game
rewards, to get better than the heuristic it started from.

```bash
# This is the exact setup used for the 73.5% win rate model
python scripts/train_warmstart.py --no-shop --made-hand-bias FULL_HOUSE,FLUSH \
    --checkpoint-path checkpoints/my_run

# If training stops or crashes partway, resume it from the last saved checkpoint
python scripts/train_warmstart.py --no-shop --init-checkpoint checkpoints/my_run_step_XXXXXX_steps.zip \
    --timesteps <how many steps are left> --checkpoint-path checkpoints/my_run
```

A few useful flags: `--timesteps` sets how long to train (default 15 million steps), `--n-envs`
sets how many games run in parallel (default 24), and `--made-hand-bias FULL_HOUSE,FLUSH` makes
the training deal a full house or flush more often on purpose. This matters because a normal
random deal almost never already contains one of these strong hands, so without this flag the
model barely ever practices recognizing and keeping a great hand it was just dealt. Training also
saves a checkpoint every 200,000 steps by default (`--checkpoint-every`), so a crash never costs
you more progress than that.

```bash
python scripts/plot_progress.py logs/my_run_log.csv          # makes a reward and ante chart
```

## Playing against the real game

`scripts/play_live.py` connects to a real running copy of Balatro through
[BalatroBot](https://github.com/coder/balatrobot), a mod that opens up the game to outside
programs. The model was never trained on the real game client, only on the simulator, so this is
a genuine test of whether it actually learned something useful.

### One time setup

1. Install [Balatro](https://store.steampowered.com/app/2379780/Balatro/) through Steam.
2. Install the [Lovely Injector](https://github.com/ethangreen-dev/lovely-injector). This lets
   mods change how the game runs.
3. Install [Steamodded](https://github.com/Steamopollys/Steamodded). This is the mod loader that
   other Balatro mods, including BalatroBot, need to work.
4. Install [BalatroBot](https://github.com/coder/balatrobot) into your Balatro mods folder
   (on Windows this is `%AppData%/Balatro/mods/`). It comes with its own Python setup. From
   inside that mod folder, run:
   ```bash
   uv sync   # or: pip install -e .
   ```

### Running it

```bash
# In one terminal, from the balatrobot mod folder, this opens Balatro itself with the mod
# active, and starts a small server on 127.0.0.1:12346 that our script talks to
cd path/to/mods/balatrobot
uv run balatrobot serve --fast            # --fast runs the game at 10x speed
# add --headless if you don't want the game window to actually show up on screen

# In a second terminal, from this repo, once you see "Balatro running on port 12346":
python scripts/play_live.py --checkpoint checkpoints/stable/<your-model>.zip \
    --num-games 20 --max-ante 1
```

While it plays, the script prints out every hand it is dealt, what it decides to do, and every
shop purchase, so you can follow along:

```
[1] hand: 0:AS 1:5H 2:4H 3:3S 4:2D  (best available: STRAIGHT)
[1] SELECTING_HAND -> play [0, 2, 4, 5, 7] (AS 5H 4H 3S 2D)
```

Shop purchases are handled by a simple fixed rule, not by the trained model itself. It buys the
cheapest joker it can afford if there is a free slot, and otherwise moves on. The model was only
ever trained to make card decisions, so it makes sense to keep it out of shop decisions it never
learned.

One thing to know: the Balatro game process itself can sometimes crash during long runs of many
games in a row. This is a stability issue with the game engine, not a bug in this repo's code.
`play_live.py` automatically retries a failed request a few times, and if the game really has
crashed, it prints a clean summary of everything completed so far and stops cleanly. Just start
the server again with the same two commands above and run the script again for however many
games are left.

## How the AI sees the game and makes decisions

What it sees: each of the 8 cards in hand (rank and suit), how many hands and discards are left,
how far along the round is, current money, the current ante and blind, the best poker hand type
it could currently make, and which exact cards make up that best hand.

What it can do: for each of the 8 cards in hand, it decides yes or no on whether to include that
card, plus one more decision on whether it is playing or discarding those cards. If it ever picks
something invalid, like 0 cards or more than 5, the game automatically fixes it into a valid move
instead of just punishing it, so every single decision it makes actually does something.

How it gets rewarded: on every hand it plays, it earns a reward based on how much of the blind's
required score that hand covered, plus a bonus for how close its play was to the best possible
hand it could have made with those same cards (checked by trying every possible combination). That
second part is what actually teaches it to always play its strongest hand instead of a weaker one.
On top of that there are small bonuses for beating a blind and winning the run, a small cost for
discarding, and a penalty for losing.

## The bug that took the longest to fix

For a long time, the AI kept breaking apart a full house or flush it had already been dealt,
playing something much weaker instead. This showed up both in the simulator and in real games.

It turned out the real problem was not the reward, it was how the AI was allowed to describe
which cards it wanted to play. The first version asked it to pick one number out of 218 possible
numbers, where each number secretly stood for a specific group of cards. That is an extremely
hard thing to learn, since two nearby numbers have nothing in common with each other card wise.

Switching to a simple yes or no per card seemed like it should be easier, but on its own it just
made the AI say yes to almost every card. What actually fixed it was adding a direct hint to what
the AI sees: for each card, telling it plainly whether that card is part of the best hand
currently available. Once the AI could directly compare "should I flag this card" against "is
this card part of the best hand," it learned to copy that signal properly.

After this fix, correctly keeping a full house or flush together went from under 10% of the time
to somewhere around 85 to 95% in testing, and that held up through a full training run and 200
real games played against the live game.
