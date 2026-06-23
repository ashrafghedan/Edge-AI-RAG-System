from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from edge_rag.types import ActiveCorpus, GeneratedQuestion, SourceText
from edge_rag.utils import normalize_whitespace

from .config import get_settings
from .database import get_db
from .models import (
    AppSessionRecord,
    AuthTokenRecord,
    CorpusDocumentRecord,
    CorpusRecord,
    DocumentRecord,
    GeneratedQuestionRecord,
    UserRecord,
    utc_now,
)
from .runtime import SessionRuntimeManager
from .schemas import ActiveCorpusResponse, UserResponse


_bearer_scheme = HTTPBearer(auto_error=False)


def event_id() -> str:
    return uuid4().hex


def active_corpus_response(
    active_corpus: ActiveCorpus,
    *,
    corpus_id: str | None = None,
    document_ids: list[str] | None = None,
) -> ActiveCorpusResponse:
    return ActiveCorpusResponse(
        corpus_id=corpus_id,
        dataset_id=active_corpus.dataset_id,
        dataset_label=active_corpus.dataset_label,
        source_names=active_corpus.source_names,
        source_paths=active_corpus.source_paths,
        chunk_count=active_corpus.chunk_count,
        vector_directory=str(active_corpus.vector_directory),
        document_ids=document_ids or [],
    )


def corpus_record_response(corpus: CorpusRecord, document_ids: list[str]) -> ActiveCorpusResponse:
    return ActiveCorpusResponse(
        corpus_id=corpus.id,
        dataset_id=corpus.dataset_id,
        dataset_label=corpus.dataset_label,
        source_names=list(corpus.source_names or []),
        source_paths=list(corpus.source_paths or []),
        chunk_count=corpus.chunk_count,
        vector_directory=corpus.vector_directory,
        document_ids=document_ids,
    )


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if '@' not in normalized or normalized.startswith('@') or normalized.endswith('@'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Enter a valid email address.')
    return normalized


def normalize_display_name(display_name: str | None, email: str) -> str:
    candidate = (display_name or '').strip()
    if candidate:
        return candidate[:255]
    local_part = email.split('@', 1)[0].replace('.', ' ').replace('_', ' ').replace('-', ' ')
    compact = ' '.join(part for part in local_part.split() if part)
    return (compact.title() or 'User')[:255]


def user_initials(display_name: str) -> str:
    parts = [part[:1].upper() for part in display_name.split() if part]
    if not parts:
        return 'U'
    if len(parts) == 1:
        return parts[0]
    return ''.join(parts[:2])


def user_response(user: UserRecord) -> UserResponse:
    return UserResponse(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        initials=user_initials(user.display_name),
        created_at=user.created_at,
    )


def hash_password(password: str) -> str:
    cleaned = password.strip()
    if len(cleaned) < 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Password must be at least 8 characters.')
    settings = get_settings()
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        'sha256',
        cleaned.encode('utf-8'),
        salt.encode('utf-8'),
        settings.password_pbkdf2_iterations,
    )
    return f'pbkdf2_sha256${settings.password_pbkdf2_iterations}${salt}${digest.hex()}'


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt, expected = stored_hash.split('$', 3)
    except ValueError:
        return False
    if algorithm != 'pbkdf2_sha256':
        return False
    candidate = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        int(iterations_text),
    ).hex()
    return hmac.compare_digest(candidate, expected)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def issue_access_token(db: Session, user: UserRecord) -> str:
    settings = get_settings()
    token = secrets.token_urlsafe(48)
    db.add(
        AuthTokenRecord(
            id=uuid4().hex,
            user_id=user.id,
            token_hash=_hash_token(token),
            expires_at=utc_now() + timedelta(days=settings.auth_token_ttl_days),
        )
    )
    return token


def get_current_token_record(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> AuthTokenRecord:
    if credentials is None or credentials.scheme.lower() != 'bearer' or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required.')

    token_hash = _hash_token(credentials.credentials)
    record = db.scalar(select(AuthTokenRecord).where(AuthTokenRecord.token_hash == token_hash))
    if record is None or _coerce_utc(record.expires_at) <= utc_now():
        if record is not None:
            db.execute(delete(AuthTokenRecord).where(AuthTokenRecord.id == record.id))
            db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required.')
    return record


def get_current_user(
    token: AuthTokenRecord = Depends(get_current_token_record),
    db: Session = Depends(get_db),
) -> UserRecord:
    user = db.get(UserRecord, token.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required.')
    return user


def get_session_or_404(
    session_id: str,
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AppSessionRecord:
    session = db.scalar(
        select(AppSessionRecord).where(
            AppSessionRecord.id == session_id,
            AppSessionRecord.user_id == current_user.id,
            AppSessionRecord.is_library.is_(False),
        )
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Session not found.')
    return session


def get_or_create_library_session(db: Session, user: UserRecord) -> AppSessionRecord:
    record = db.scalar(
        select(AppSessionRecord).where(
            AppSessionRecord.user_id == user.id,
            AppSessionRecord.is_library.is_(True),
        )
    )
    if record is not None:
        return record

    record = AppSessionRecord(
        id=uuid4().hex,
        user_id=user.id,
        title='Document Library',
        is_library=True,
    )
    db.add(record)
    db.flush()
    return record


def ensure_runtime_active_corpus(
    session_id: str,
    db: Session,
    runtime_manager: SessionRuntimeManager,
) -> tuple[ActiveCorpus, CorpusRecord, list[str]]:
    session = db.get(AppSessionRecord, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Session not found.')
    if not session.active_corpus_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No active corpus for this session.')

    corpus = db.get(CorpusRecord, session.active_corpus_id)
    if corpus is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Active corpus metadata is missing.')

    document_ids = db.scalars(
        select(CorpusDocumentRecord.document_id).where(CorpusDocumentRecord.corpus_id == corpus.id)
    ).all()
    documents = db.scalars(
        select(DocumentRecord).where(DocumentRecord.id.in_(document_ids)).order_by(DocumentRecord.uploaded_at.asc())
    ).all()
    if not documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Active corpus has no documents.')

    sources = [source_text_from_document_record(record) for record in documents]
    paths = [Path(record.storage_path) for record in documents]
    app = runtime_manager.get_app(session_id)
    try:
        active = app.require_active_corpus()
        current_paths = {Path(path).resolve() for path in active.source_paths}
        target_paths = {path.resolve() for path in paths}
        if current_paths != target_paths:
            active = runtime_manager.activate_sources(session_id, sources)
    except RuntimeError:
        active = runtime_manager.activate_sources(session_id, sources)
    return active, corpus, document_ids


def restore_generated_question(record: GeneratedQuestionRecord) -> GeneratedQuestion:
    created_at = record.created_at.isoformat()
    return GeneratedQuestion(
        question_id=record.id,
        question=normalize_whitespace(record.question_text),
        model_answer=normalize_whitespace(record.model_answer),
        source_names=list(record.source_names or []),
        source_chunk_ids=list(record.source_chunk_ids or []),
        created_at=created_at,
    )


def source_text_from_document_record(record: DocumentRecord) -> SourceText:
    return SourceText(
        name=record.original_name,
        path=Path(record.storage_path).resolve(),
        content=record.content_text,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        modified_at=str(record.modified_at.timestamp()),
    )
