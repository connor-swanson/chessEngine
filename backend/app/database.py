import os
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# data/chess.db at the repo root, independent of the working directory.
# CHESS_DB_URL overrides it (the tests use a temporary database).
_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "chess.db"
DATABASE_URL = os.environ.get("CHESS_DB_URL", f"sqlite+aiosqlite:///{_DEFAULT_DB}")

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migration: add the source column to existing databases.
        # SQLite raises an error if the column already exists; we swallow it.
        try:
            await conn.execute(text("ALTER TABLE games ADD COLUMN source VARCHAR(20)"))
        except Exception:
            pass  # column already present — nothing to do
