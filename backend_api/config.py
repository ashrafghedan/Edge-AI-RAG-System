from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from edge_rag.env import load_project_env


@dataclass(frozen=True, slots=True)
class ApiSettings:
    project_root: Path
    api_prefix: str
    database_url: str
    uploads_dir: Path
    runtime_dir: Path
    cors_origins: list[str]
    chat_history_limit: int
    auth_token_ttl_days: int
    password_pbkdf2_iterations: int
    stt_backend: str
    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_default_language: str | None
    stt_max_file_mb: int
    pdf_upload_max_figures: int
    pdf_upload_describe_pages: bool


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


@lru_cache
def get_settings() -> ApiSettings:
    project_root = Path(__file__).resolve().parent.parent
    load_project_env(project_root)
    data_dir = project_root / 'data'
    database_url = os.getenv('DATABASE_URL', _sqlite_url(data_dir / 'app.db'))
    cors_origins = [
        value.strip()
        for value in os.getenv('FRONTEND_ORIGINS', 'http://localhost:5173').split(',')
        if value.strip()
    ]
    return ApiSettings(
        project_root=project_root,
        api_prefix='/api/v1',
        database_url=database_url,
        uploads_dir=data_dir / 'uploads',
        runtime_dir=data_dir / 'api_runtime',
        cors_origins=cors_origins,
        chat_history_limit=int(os.getenv('CHAT_HISTORY_LIMIT', '10')),
        auth_token_ttl_days=int(os.getenv('AUTH_TOKEN_TTL_DAYS', '30')),
        password_pbkdf2_iterations=int(os.getenv('PASSWORD_PBKDF2_ITERATIONS', '600000')),
        stt_backend=os.getenv('EDGE_RAG_STT_BACKEND', 'faster_whisper').strip().lower() or 'faster_whisper',
        stt_model=os.getenv('EDGE_RAG_STT_MODEL', 'data/models/faster-whisper-tiny').strip()
        or 'data/models/faster-whisper-tiny',
        stt_device=os.getenv('EDGE_RAG_STT_DEVICE', 'auto').strip().lower() or 'auto',
        stt_compute_type=os.getenv('EDGE_RAG_STT_COMPUTE_TYPE', 'int8').strip(),
        stt_default_language=os.getenv('EDGE_RAG_STT_LANGUAGE', '').strip().lower() or None,
        stt_max_file_mb=int(os.getenv('EDGE_RAG_STT_MAX_FILE_MB', '25')),
        pdf_upload_max_figures=int(os.getenv('CNV_PDF_UPLOAD_MAX_FIGURES', '4')),
        pdf_upload_describe_pages=os.getenv('CNV_PDF_UPLOAD_DESCRIBE_PAGES', '1').strip().lower()
        in {'1', 'true', 'yes', 'on'},
    )
