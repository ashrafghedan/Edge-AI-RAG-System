from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


settings = get_settings()
connect_args = {'check_same_thread': False} if settings.database_url.startswith('sqlite') else {}
engine = create_engine(settings.database_url, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _run_schema_migrations()


def _run_schema_migrations() -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if 'app_sessions' in tables:
        _ensure_columns(
            'app_sessions',
            {
                'user_id': _ddl_by_dialect(
                    postgres='ALTER TABLE app_sessions ADD COLUMN IF NOT EXISTS user_id VARCHAR(64)',
                    sqlite='ALTER TABLE app_sessions ADD COLUMN user_id VARCHAR(64)',
                ),
                'is_library': _ddl_by_dialect(
                    postgres='ALTER TABLE app_sessions ADD COLUMN IF NOT EXISTS is_library BOOLEAN NOT NULL DEFAULT FALSE',
                    sqlite='ALTER TABLE app_sessions ADD COLUMN is_library BOOLEAN NOT NULL DEFAULT 0',
                ),
            },
        )
    if 'documents' in tables:
        _ensure_columns(
            'documents',
            {
                'user_id': _ddl_by_dialect(
                    postgres='ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id VARCHAR(64)',
                    sqlite='ALTER TABLE documents ADD COLUMN user_id VARCHAR(64)',
                ),
                'content_text': _ddl_by_dialect(
                    postgres="ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_text TEXT NOT NULL DEFAULT ''",
                    sqlite="ALTER TABLE documents ADD COLUMN content_text TEXT NOT NULL DEFAULT ''",
                ),
            },
        )
    if 'auth_tokens' in tables:
        _ensure_columns(
            'auth_tokens',
            {
                'expires_at': _ddl_by_dialect(
                    postgres='ALTER TABLE auth_tokens ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ',
                    sqlite='ALTER TABLE auth_tokens ADD COLUMN expires_at TIMESTAMP',
                ),
            },
        )
    _backfill_flags()


def _ensure_columns(table_name: str, statements: dict[str, str]) -> None:
    inspector = inspect(engine)
    existing_columns = {column['name'] for column in inspector.get_columns(table_name)}
    dialect = engine.dialect.name
    with engine.begin() as conn:
        for column_name, statement in statements.items():
            if column_name in existing_columns:
                continue
            if dialect == 'sqlite':
                conn.exec_driver_sql(statement)
            else:
                conn.execute(text(statement))


def _backfill_flags() -> None:
    dialect = engine.dialect.name
    with engine.begin() as conn:
        if dialect == 'sqlite':
            conn.exec_driver_sql('UPDATE app_sessions SET is_library = 0 WHERE is_library IS NULL')
        else:
            conn.execute(text('UPDATE app_sessions SET is_library = FALSE WHERE is_library IS NULL'))


def _ddl_by_dialect(*, postgres: str, sqlite: str) -> str:
    return postgres if engine.dialect.name == 'postgresql' else sqlite
