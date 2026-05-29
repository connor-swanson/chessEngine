import chess
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..database import get_db
from ..models import Game, Position
from ..schemas import CreateGameRequest, GameStateResponse, GameSummary, PositionResponse
from ..game_manager import game_manager
from ..chess_utils import board_to_state, fen_to_tensor

router = APIRouter(prefix="/games", tags=["games"])


@router.post("", response_model=GameStateResponse)
async def create_game(req: CreateGameRequest, db: AsyncSession = Depends(get_db)):
    game = Game(white_player=req.white_player, black_player=req.black_player, source="manual")
    db.add(game)
    await db.commit()
    await db.refresh(game)

    board = game_manager.create(game.id)

    db.add(Position(game_id=game.id, move_number=0, fen=board.fen(), move_uci=None, is_terminal=False))
    await db.commit()

    return {**board_to_state(board, game.id), "white_player": game.white_player, "black_player": game.black_player}


@router.get("", response_model=list[GameSummary])
async def list_games(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Game, func.count(Position.id).label("pos_count"))
        .outerjoin(Position, Position.game_id == Game.id)
        .group_by(Game.id)
        .order_by(Game.created_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [
        GameSummary(
            id=g.id,
            white_player=g.white_player,
            black_player=g.black_player,
            result=g.result,
            created_at=g.created_at,
            move_count=max(0, pos_count - 1),
            source=g.source,
        )
        for g, pos_count in rows
    ]


@router.get("/{game_id}", response_model=GameStateResponse)
async def get_game(game_id: int, db: AsyncSession = Depends(get_db)):
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

    return {**board_to_state(board, game_id), "white_player": game.white_player, "black_player": game.black_player}


@router.get("/{game_id}/positions", response_model=list[PositionResponse])
async def get_positions(game_id: int, db: AsyncSession = Depends(get_db)):
    if not await db.get(Game, game_id):
        raise HTTPException(status_code=404, detail="Game not found")

    positions = (await db.execute(
        select(Position)
        .where(Position.game_id == game_id)
        .order_by(Position.move_number)
    )).scalars().all()

    result = []
    for pos in positions:
        tensor = fen_to_tensor(pos.fen)
        result.append(PositionResponse(
            id=pos.id,
            move_number=pos.move_number,
            fen=pos.fen,
            move_uci=pos.move_uci,
            is_terminal=pos.is_terminal,
            tensor_shape=list(tensor.shape),
            tensor=tensor.tolist(),
        ))
    return result
