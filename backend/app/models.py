from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    white_player: Mapped[str] = mapped_column(String(50))
    black_player: Mapped[str] = mapped_column(String(50))
    result: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    pgn: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    positions: Mapped[list["Position"]] = relationship("Position", back_populates="game")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id"))
    move_number: Mapped[int] = mapped_column(Integer)
    fen: Mapped[str] = mapped_column(String(150))
    move_uci: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    is_terminal: Mapped[bool] = mapped_column(Boolean, default=False)

    game: Mapped["Game"] = relationship("Game", back_populates="positions")
