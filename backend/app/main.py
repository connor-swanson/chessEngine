from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db, SessionLocal
from .models import Game
from .routers import games, moves, simulations
from .game_manager import game_manager
from .chess_utils import board_to_state
from .websocket_manager import connection_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Chess Engine API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(games.router)
app.include_router(moves.router)
app.include_router(simulations.router)


@app.websocket("/games/{game_id}/ws")
async def websocket_endpoint(websocket: WebSocket, game_id: int):
    await connection_manager.connect(game_id, websocket)
    try:
        # Push current state immediately on connect
        board = game_manager.get(game_id)
        if board:
            async with SessionLocal() as db:
                game = await db.get(Game, game_id)
                if game:
                    state = {
                        **board_to_state(board, game_id),
                        "white_player": game.white_player,
                        "black_player": game.black_player,
                    }
                    await websocket.send_json(state)

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(game_id, websocket)
