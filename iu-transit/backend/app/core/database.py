"""
app/core/database.py
────────────────────
Async SQLAlchemy engine for SQLite with WAL mode enabled.
aiosqlite uses NullPool by default — pool_size/max_overflow are not valid.
WAL mode is set via a connection event listener on every new connection.
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event, text
from app.core.config import settings

os.makedirs("./data", exist_ok=True)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,        # wait up to 30s for a write lock before failing
    },
)


def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """
    Enable WAL journal mode on every new SQLite connection.
    WAL allows concurrent reads while writing and serialises writes
    gracefully instead of throwing "database is locked" immediately.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=15000")   # 15s busy wait at SQLite level
    cursor.close()


event.listen(engine.sync_engine, "connect", _set_sqlite_pragmas)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables on startup (idempotent)."""
    async with engine.begin() as conn:
        from app.models import gtfs, schedule, weather, predictions  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a session, commits on success, rolls back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
