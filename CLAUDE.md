# Chess Engine — RL Training Project

## Goal

Train a chess engine using Reinforcement Learning — four educational notebooks (SFT → REINFORCE → PPO → AlphaZero MCTS) that end in an engine of ~1480 ± 75 Elo (vs Stockfish UCI_Elo 1320 at 0.1 s/move). Three phases, each building on the last.

---

## Phase 1 — Chess App

Build a Python backend + React frontend for playing chess games.

**Backend** (`backend/`) — FastAPI + python-chess + SQLite
- `app/main.py` — FastAPI entry point, WebSocket endpoint, CORS
- `app/database.py` — SQLAlchemy async engine (aiosqlite)
- `app/models.py` — `Game` and `Position` ORM models
- `app/schemas.py` — Pydantic request/response schemas
- `app/game_manager.py` — In-memory dict of active `chess.Board` instances
- `app/chess_utils.py` — FEN helpers + `fen_to_tensor()` → `(14, 8, 8)` numpy array
- `app/bots/` — `Bot` abstract base + registry (`BOT_FACTORIES`/`make_bot`/`get_bot`). Bots: `random`, `heuristic` (+`-e10/e20/e35` epsilon variants), `ab1/ab2/ab3` (alpha-beta, + `ab2-e10/e25`), `stockfish*`, and the trained neural engine: `engine` (200 sims MCTS), `engine-fast` (50), `engine-policy` (raw policy), `sft`, `mcts` (`neural_bot.py`, lazy torch import; checkpoints in `train/checkpoints/*.pt`, override dir with `CHESS_CHECKPOINT_DIR`). Neural bots error when used if torch or the checkpoint is missing.
- `app/routers/games.py` — Game CRUD endpoints
- `app/routers/moves.py` — Move submission + bot-move endpoints
- `app/routers/simulations.py` — Bulk game simulation endpoints
- `app/routers/bots.py` — `GET /bots` lists bots + availability; the frontend dropdowns use it (`hooks/useBots.ts`)

**Frontend** (`frontend/`) — React + Vite + react-chessboard
- Drag-and-drop board, auto-queening promotion
- Live board updates via WebSocket
- Bot move auto-play loop
- Simulations tab for running bulk games

**Run:**
```bash
# Backend
cd backend && .venv/bin/uvicorn app.main:app --reload

# Frontend
cd frontend && npm run dev
```

Neural bots need torch in the backend venv (`backend/.venv/bin/pip install torch==2.8.0`, listed as optional in `requirements.txt`) and a checkpoint such as `train/checkpoints/engine.pt`; otherwise they are hidden from the UI.

**Key API endpoints:**
| Method | Path | Description |
|---|---|---|
| POST | `/games` | Create game |
| GET | `/games/{id}` | Get game state |
| POST | `/games/{id}/move` | Submit move (UCI) |
| POST | `/games/{id}/bot-move` | Trigger bot move |
| GET | `/bots` | Bot names + availability |
| GET | `/games/{id}/positions` | Full position history + tensors |
| WS | `/games/{id}/ws` | Live board state push |

The backend's `chess_utils.fen_to_tensor()` (14 planes) only serves the `/positions` endpoint; training uses `train/chesslib/encoding.py` (below).

---

## Phase 2 — Game Simulation (data)

- `train/generate_games.py` — parallel bot-vs-bot generator writing to `data/chess.db` (same schema as the backend, `source='simulation'`). Example: `train/.venv/bin/python train/generate_games.py --white ab2 --black ab2 --games 6000 --workers 8 --random-opening-plies 6`.
- `positions` rows hold the FEN **after** `move_uci`; never pair a row's FEN with its own move. `chesslib.data.load_games/build_dataset` replay move lists instead.
- Bot strength on our scale (anchored on Stockfish `UCI_Elo=1320` at 0.1 s/move, ±~140): ab1 ≈ 1200, ab2 ≈ 1470, ab3 ≈ 1670. `heuristic` (1-ply material) loses 0–30 to ab1.

---

## Phase 3 — Training notebooks (`train/`)

Run all four in order with `cd train && ./run_pipeline.sh` (≈2.5 h on an M4, plus ~1 h the first time to simulate the teacher games; outputs saved in place). Each loads `checkpoints/sft.pt` and writes `checkpoints/<name>.pt` + `results/<name>.json`.

| Notebook | Method | Output |
|---|---|---|
| `chess_sft.ipynb` | behavioural cloning of `ab2` (simulates 6k games if missing) | `sft.pt` |
| `chess_reinforce.ipynb` | REINFORCE with value baseline | `reinforce.pt` |
| `chess_ppo.ipynb` | PPO (clipped ratio, GAE) | `ppo.pt` |
| `chess_mcts.ipynb` | AlphaZero-style MCTS self-play + gating, Elo eval | `mcts.pt`, **`engine.pt`** |

`train/chesslib/` — shared plumbing (notebooks keep the algorithm itself in the notebook):
- `encoding.py` — 21 planes from the side-to-move's view (mirrored for Black; repetition, castling, en-passant, 50-move planes); `ACTION_SIZE = 4168` (`from*64+to` + 72 under-promotions)
- `model.py` — `ChessNet` ResNet (default 6×64), dot-product policy head, WDL value head; `save_checkpoint`/`load_checkpoint`
- `data.py` — DB → arrays, split by game, batches with legal masks
- `mcts.py` — batched PUCT search (many trees, one network batch), tree reuse; `selfplay.py` — self-play workers (MPS) + replay buffer
- `rollouts.py` — batched policy rollouts for REINFORCE/PPO; `arena.py` — players, paired-random-opening matches, Elo fit

Tests: `train/.venv/bin/python -m pytest train/tests backend/tests -q`.

## Notes

- Python 3.9 — use `Optional[X]` not `X | None`, add `from __future__ import annotations` for forward refs
- SQLite lives in `data/chess.db` (gitignored)
- CORS is open (`allow_origins=["*"]`) for local dev — lock down before any deployment
- `greenlet` is a required SQLAlchemy async dep not pulled in automatically on Python 3.9 — it's in `requirements.txt`
