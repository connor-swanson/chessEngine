# Chess Engine — learning RL by building one

A playable chess app plus four Jupyter notebooks that train a neural chess
engine step by step, each one teaching a different learning method:

1. **Supervised learning** — copy a scripted bot's moves
2. **REINFORCE** — policy gradient from game results
3. **PPO** — actor-critic policy gradient with a trust region
4. **AlphaZero-style MCTS** — tree search plus self-play

The pipeline runs end to end: simulate bot games → train → evaluate → play the
engine in the browser.

## Results

All ratings are on one scale, anchored to Stockfish's `UCI_Elo=1320` setting at
0.1 s per move. Treat them as roughly ±100. Stockfish's weakening is calibrated
at longer time controls, so these numbers probably flatter the engine compared
with a human rating.

| Player | Elo | Notes |
|---|---|---|
| `ab1` / `ab2` / `ab3` (scripted alpha-beta bots) | ≈1200 / 1470 / 1670 | ±~140 |
| SFT network, no search | ≈920 | 15–21% vs ab1 |
| **Final engine** (`engine.pt`, 200-sim MCTS) | **≈1480 ± 75** | 69-5-26 vs Stockfish-1320, 85% vs ab1, 81% vs ab2 |

What each method contributed, each network playing without search:

| Network | Score vs ab1 (no search) | Takeaway |
|---|---|---|
| SFT | 15% | dense, cheap supervision; capped by the teacher |
| REINFORCE | 23% | stable, but within noise of SFT |
| PPO | 24% | sampled play improved; best move barely changed |
| AlphaZero network | 27% | +53 ± 70 Elo over SFT (both with search) |
| **AlphaZero network + 200-sim search** | **85%** | search does most of the work |

The exact numbers are in `train/results/*.json` and in the saved notebook outputs.

## Repository layout

```
backend/            FastAPI + python-chess + SQLite: game API, WebSocket, bots
  app/bots/         random, heuristic, ab1-3 (alpha-beta), stockfish*, engine* (neural)
  app/routers/      games, moves, simulations, bots (GET /bots)
  tests/            bot + engine-bot tests
frontend/           React + Vite UI: play against any bot, run bulk simulations
train/
  chess_sft.ipynb        1 · behavioural cloning of ab2
  chess_reinforce.ipynb  2 · REINFORCE with a value baseline
  chess_ppo.ipynb        3 · PPO (clipped ratio, GAE)
  chess_mcts.ipynb       4 · AlphaZero loop, final engine + Elo evaluation
  chesslib/              shared plumbing used by every notebook
  generate_games.py      parallel bot-vs-bot game generator → data/chess.db
  run_pipeline.sh        executes all four notebooks in order
  tests/                 chesslib tests
  checkpoints/           *.pt (gitignored)   results/  per-notebook JSON metrics
data/chess.db       all games, both played and simulated (gitignored)
```

## How it works

```
 bots (ab2 vs ab2)  ──generate_games.py──▶  data/chess.db
                                                │  replay move lists
                                                ▼
                      1 SFT: imitate ab2  ──▶  sft.pt ──┬──▶ 2 REINFORCE ─▶ reinforce.pt
                                                        ├──▶ 3 PPO       ─▶ ppo.pt
                                                        └──▶ 4 AlphaZero ─▶ mcts.pt ─▶ engine.pt
                                                                                        │
                                              backend "engine" bot (MCTS)  ◀────────────┘
```

**Data.** The `games` and `positions` tables store every game, one position
row per ply. A position row holds the board *after* its move. Pairing a row's
board with that row's own move would teach the network to guess the previous
move, so `chesslib.data` replays each game's move list instead. Datasets are
split into train and validation **by game**, because neighbouring positions
from one game are near-duplicates.

**Board encoding** (`chesslib/encoding.py`). The board becomes 21 planes of
8×8, always from the point of view of the side to move: for Black the board is
mirrored and the colours swapped. The planes cover each piece type for both
sides, repetition counters, castling rights, the en-passant square, the 50-move
counter and a constant plane. Moves become one of 4,168 actions: `from*64 + to`,
plus 72 slots for under-promotions. The network only ever chooses among legal
moves.

**Network** (`chesslib/model.py`). A small ResNet (6 blocks × 64 filters, about
0.56M parameters) with two heads:
- a policy head that scores each *from-square → to-square* pair by a dot
  product of per-square embeddings;
- a value head that predicts win/draw/loss.

It's kept small on purpose, because the search evaluates it tens of thousands
of times per second.

**Search** (`chesslib/mcts.py`). PUCT Monte-Carlo tree search, with:
- mates and draws scored exactly;
- repeated positions inside the tree treated as draws;
- pessimistic values for unvisited moves ("first-play urgency");
- tree reuse between moves.

Many games search at once, and the positions waiting to be scored from all of
them go to the network in one batch. On Apple-silicon MPS that is about 20×
faster than scoring one position at a time on a CPU core.

**Self-play** (`chesslib/selfplay.py`). Each move adds Dirichlet noise at the
root and picks moves at random for the first plies, weighted by visit count.
Games end early when the engine resigns a lost position; resignation is
switched off in 20% of games to check how often it would have been wrong. Each
iteration then:
1. trains on a replay buffer, half self-play targets (π, z) and half teacher data;
2. plays the new network against the current best;
3. promotes it only if it scores over 55%.

**Evaluation** (`chesslib/arena.py`). Every match starts from paired random
openings: each opening is played twice with colours swapped, so games differ
and the white-move advantage cancels out. Players can be random, any backend
bot, Stockfish at a fixed Elo, a raw network policy, or a network with search.
`fit_elo` turns match results against rated opponents into a rating with a
confidence interval.

## Setup

Requires Python 3.9+, Node 20.19+ (for Vite 8), and Stockfish on `PATH` (`brew install
stockfish`). Stockfish is only used as a rated opponent in evaluation.

```bash
# training environment
cd train && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..

# backend (installs torch too; it's only needed for the neural engine bots)
cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..

# frontend
cd frontend && npm install && cd ..
```

## Train the engine

```bash
cd train && ./run_pipeline.sh
```

This runs all four notebooks in place and saves their outputs: about 2.5 hours
on an Apple M4, plus about 1 hour the first time while notebook 1 simulates
6,000 teacher games. To run a single notebook, pass its name:
`./run_pipeline.sh chess_mcts.ipynb`. You can also open the notebooks in
Jupyter and step through them. Notebooks 2–4 all start from `checkpoints/sft.pt`.

To simulate games yourself:

```bash
train/.venv/bin/python train/generate_games.py --white ab2 --black ab2 --games 1000 --workers 8 --random-opening-plies 6
```

## Play against it

```bash
cd backend && .venv/bin/uvicorn app.main:app --reload    # http://localhost:8000
cd frontend && npm run dev                               # http://localhost:5173
```

Pick an opponent in the UI:

| Bot | Search | Speed |
|---|---|---|
| `engine` | 200 simulations | about 2 s per move on CPU |
| `engine-fast` | 50 simulations | fast |
| `engine-policy` | none (raw network) | instant |

The scripted bots and Stockfish levels are listed alongside them. `GET /bots`
reports which bots are available: neural bots need torch and their checkpoint.


