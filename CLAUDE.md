# Chess Engine — RL Training Project

## Goal

Train a chess engine using Reinforcement Learning. The project is broken into three phases, each building on the last.

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
- `app/bots/` — `Bot` abstract base + `RandomBot` + `HeuristicBot` (depth-1 minimax)
- `app/routers/games.py` — Game CRUD endpoints
- `app/routers/moves.py` — Move submission + bot-move endpoints
- `app/routers/simulations.py` — Bulk game simulation endpoints

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

**Key API endpoints:**
| Method | Path | Description |
|---|---|---|
| POST | `/games` | Create game |
| GET | `/games/{id}` | Get game state |
| POST | `/games/{id}/move` | Submit move (UCI) |
| POST | `/games/{id}/bot-move` | Trigger bot move |
| GET | `/games/{id}/positions` | Full position history + tensors |
| WS | `/games/{id}/ws` | Live board state push |

**RL state representation** — `fen_to_tensor(fen)` returns `(14, 8, 8)` float32:
- Planes 0–11: piece type × color
- Plane 12: side to move
- Plane 13: castling rights

---

## Phase 2 — Self-Play Data Collection 🔜

Use the Phase 1 app to simulate thousands of games between bots and collect training data.

- Run bulk bot-vs-bot games (random, heuristic, future RL agents)
- Export position datasets from `/games/{id}/positions`
- Each record: FEN string + `(14, 8, 8)` tensor + move played + game outcome
- Store in a format suitable for batched training (e.g. HDF5 or numpy archives)

**Bot interface** — any new bot just implements:
```python
class Bot(ABC):
    async def choose_move(self, board: chess.Board) -> chess.Move: ...
```

---

## Phase 3 — RL Training 🔜

Iteratively train a neural network chess engine using collected self-play data.

- Start with supervised pre-training on Phase 2 data (predict moves/outcomes)
- Move to self-play RL loop (policy + value network, similar to AlphaZero architecture)
- Plug trained model in as a `Bot` subclass — no game loop changes needed
- Evaluate against `RandomBot` and `HeuristicBot` to track improvement

---

## Notes

- Python 3.9 — use `Optional[X]` not `X | None`, add `from __future__ import annotations` for forward refs
- SQLite lives in `data/chess.db` (gitignored)
- CORS is open (`allow_origins=["*"]`) for local dev — lock down before any deployment
- `greenlet` is a required SQLAlchemy async dep not pulled in automatically on Python 3.9 — it's in `requirements.txt`
