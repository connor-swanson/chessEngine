from __future__ import annotations

from typing import Optional

import chess
import chess.pgn
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..bots import available_bots, get_bot
from ..database import get_db
from ..game_manager import game_manager
from ..models import Game, Position
from ..schemas import GameStateResponse, HintResponse, MoveRequest, ResignRequest, UndoRequest
from ..state import game_ending, game_state
from ..websocket_manager import connection_manager

router = APIRouter(prefix="/games", tags=["moves"])

# Bots tried, in order, when /hint is called without one.
_HINT_BOTS = ("engine-fast", "ab2")


def _board_to_pgn(board: chess.Board, game: Game, termination: Optional[str] = None) -> str:
    pgn = chess.pgn.Game.from_board(board)
    pgn.headers["White"] = game.white_player
    pgn.headers["Black"] = game.black_player
    if game.result:
        pgn.headers["Result"] = game.result
    if termination:
        pgn.headers["Termination"] = termination
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return pgn.accept(exporter)


async def _load(game_id: int, db: AsyncSession) -> tuple[Game, chess.Board]:
    game = await db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    board = game_manager.get(game_id)
    if board is None:
        positions = (await db.execute(
            select(Position)
            .where(Position.game_id == game_id)
            .order_by(Position.move_number)
        )).scalars().all()
        moves = [p.move_uci for p in positions[1:] if p.move_uci]
        board = game_manager.restore(game_id, moves)
    return game, board


async def _record_move(db: AsyncSession, game: Game, board: chess.Board, move_uci: str):
    ending = game_ending(board, game)
    db.add(Position(
        game_id=game.id,
        move_number=len(board.move_stack),
        fen=board.fen(),
        move_uci=move_uci,
        is_terminal=ending is not None,
    ))
    if ending is not None:
        game.result = ending[0]
        game.pgn = _board_to_pgn(board, game, ending[1])
    await db.commit()


async def _respond(board: chess.Board, game: Game) -> dict:
    state = game_state(board, game)
    await connection_manager.broadcast(game.id, state)
    return state


@router.post("/{game_id}/move", response_model=GameStateResponse)
async def make_move(game_id: int, req: MoveRequest, db: AsyncSession = Depends(get_db)):
    game, board = await _load(game_id, db)
    if game_ending(board, game):
        raise HTTPException(status_code=400, detail="Game is already over")
    try:
        game_manager.push_move(game_id, req.move_uci)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await _record_move(db, game, board, req.move_uci)
    return await _respond(board, game)


@router.post("/{game_id}/bot-move", response_model=GameStateResponse)
async def bot_move(game_id: int, db: AsyncSession = Depends(get_db)):
    game, board = await _load(game_id, db)
    if game_ending(board, game):
        raise HTTPException(status_code=400, detail="Game is already over")

    player_type = game.white_player if board.turn == chess.WHITE else game.black_player
    bot = get_bot(player_type)
    if not bot:
        raise HTTPException(status_code=400, detail=f"'{player_type}' is not a bot player type")

    fen_before = board.fen()
    plies_before = len(board.move_stack)
    move = await bot.choose_move(board)
    # The position may have changed while the bot was thinking (takeback, resign).
    if (len(board.move_stack) != plies_before or board.fen() != fen_before
            or game_manager.get_ending(game_id)):
        raise HTTPException(status_code=409, detail="Position changed while the bot was thinking")

    game_manager.push_move(game_id, move.uci())
    await _record_move(db, game, board, move.uci())
    return await _respond(board, game)


@router.post("/{game_id}/undo", response_model=GameStateResponse)
async def undo(game_id: int, req: UndoRequest, db: AsyncSession = Depends(get_db)):
    """Take back the last `plies` half-moves (2 = your move and the bot's reply)."""
    game, board = await _load(game_id, db)
    if not board.move_stack:
        raise HTTPException(status_code=400, detail="No moves to take back")
    game_manager.undo(game_id, req.plies)
    await db.execute(delete(Position).where(
        Position.game_id == game_id, Position.move_number > len(board.move_stack)))
    game.result = None
    game.pgn = None
    await db.commit()
    return await _respond(board, game)


@router.post("/{game_id}/resign", response_model=GameStateResponse)
async def resign(game_id: int, req: ResignRequest, db: AsyncSession = Depends(get_db)):
    """End the game: `color` resigns, or loses on time (`reason="timeout"`)."""
    game, board = await _load(game_id, db)
    if game_ending(board, game):
        raise HTTPException(status_code=400, detail="Game is already over")
    loser = chess.WHITE if req.color == "white" else chess.BLACK
    result = "0-1" if loser == chess.WHITE else "1-0"
    termination = req.reason
    # A flag only loses if the opponent could still mate.
    if req.reason == "timeout" and board.has_insufficient_material(not loser):
        result, termination = "1/2-1/2", "timeout vs insufficient material"
    game_manager.set_ending(game_id, result, termination)
    game.result = result
    game.pgn = _board_to_pgn(board, game, termination)
    await db.commit()
    return await _respond(board, game)


@router.get("/{game_id}/hint", response_model=HintResponse)
async def hint(game_id: int, bot: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """Suggest a move for the side to move without playing it."""
    game, board = await _load(game_id, db)
    if game_ending(board, game):
        raise HTTPException(status_code=400, detail="Game is already over")
    if bot is None:
        avail = available_bots()
        bot = next((b for b in _HINT_BOTS if avail.get(b, {}).get("available")), "ab2")
    helper = get_bot(bot)
    if helper is None:
        raise HTTPException(status_code=400, detail=f"Unknown bot '{bot}'")
    move = await helper.choose_move(board.copy())
    return {"move_uci": move.uci(), "bot": bot}
