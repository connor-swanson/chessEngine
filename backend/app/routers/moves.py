import chess
import chess.pgn
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database import get_db
from ..models import Game, Position
from ..schemas import MoveRequest, GameStateResponse
from ..game_manager import game_manager
from ..chess_utils import board_to_state
from ..bots import get_bot
from ..websocket_manager import connection_manager

router = APIRouter(prefix="/games", tags=["moves"])


def _board_to_pgn(board: chess.Board) -> str:
    game = chess.pgn.Game.from_board(board)
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


async def _ensure_board(game_id: int, db: AsyncSession) -> chess.Board:
    board = game_manager.get(game_id)
    if board is not None:
        return board
    positions = (await db.execute(
        select(Position)
        .where(Position.game_id == game_id)
        .order_by(Position.move_number)
    )).scalars().all()
    moves = [p.move_uci for p in positions[1:] if p.move_uci]
    return game_manager.restore(game_id, moves)


async def _record_move(db: AsyncSession, game_id: int, board: chess.Board, move_uci: str):
    is_terminal = board.is_game_over()
    db.add(Position(
        game_id=game_id,
        move_number=len(board.move_stack),
        fen=board.fen(),
        move_uci=move_uci,
        is_terminal=is_terminal,
    ))
    if is_terminal:
        game = await db.get(Game, game_id)
        game.result = board.result()
        game.pgn = _board_to_pgn(board)
    await db.commit()


@router.post("/{game_id}/move", response_model=GameStateResponse)
async def make_move(game_id: int, req: MoveRequest, db: AsyncSession = Depends(get_db)):
    game = await db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    board = await _ensure_board(game_id, db)

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    try:
        game_manager.push_move(game_id, req.move_uci)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await _record_move(db, game_id, board, req.move_uci)

    state = {**board_to_state(board, game_id), "white_player": game.white_player, "black_player": game.black_player}
    await connection_manager.broadcast(game_id, state)
    return state


@router.post("/{game_id}/bot-move", response_model=GameStateResponse)
async def bot_move(game_id: int, db: AsyncSession = Depends(get_db)):
    game = await db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    board = await _ensure_board(game_id, db)

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    player_type = game.white_player if board.turn == chess.WHITE else game.black_player
    bot = get_bot(player_type)
    if not bot:
        raise HTTPException(status_code=400, detail=f"'{player_type}' is not a bot player type")

    move = await bot.choose_move(board)
    move_uci = move.uci()
    game_manager.push_move(game_id, move_uci)

    await _record_move(db, game_id, board, move_uci)

    state = {**board_to_state(board, game_id), "white_player": game.white_player, "black_player": game.black_player}
    await connection_manager.broadcast(game_id, state)
    return state
