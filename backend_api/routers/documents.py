from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from edge_rag.chunking import clear_document_chunk_status, document_chunk_status
from edge_rag.config import default_config
from edge_rag.llm_client import LlamaCppInferenceError
from edge_rag.loaders import load_sources
from edge_rag.utils import safe_label

from ..config import get_settings
from ..database import get_db
from ..deps import (
    active_corpus_response,
    event_id,
    get_current_user,
    get_or_create_library_session,
    get_session_or_404,
    source_text_from_document_record,
)
from ..models import (
    AnswerAttemptRecord,
    AppSessionRecord,
    CorpusDocumentRecord,
    CorpusRecord,
    DocumentRecord,
    GeneratedQuestionRecord,
    SessionEventRecord,
    UserRecord,
    utc_now,
)
from ..runtime import get_runtime_manager
from ..schemas import ActivateCorpusRequest, ActiveCorpusResponse, DocumentResponse, UploadDocumentsResponse


router = APIRouter(prefix='/sessions/{session_id}', tags=['documents'])


def _modified_at_from_source(value: str) -> datetime:
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _convert_pdf_to_text(pdf_path: Path) -> Path:
    settings = get_settings()
    script_path = settings.project_root / 'cnv_pdf.py'
    if not script_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='PDF conversion script was not found.',
        )

    # Use the same interpreter that runs the backend (the project venv) so the
    # converter has access to pymupdf et al. The old hardcoded '/usr/bin/python3'
    # only exists on Linux and made every PDF upload fail with a 500 on Windows.
    command = [sys.executable, str(script_path), str(pdf_path)]

    # The figure/page VLM analysis needs a vision-capable model. When vision is
    # disabled (no mmproj loaded), running it would fire image requests at a
    # text-only model and slow every upload to a crawl, so fall back to plain
    # pymupdf + Tesseract text extraction (--max-figures 0 disables the VLM).
    vision_enabled = os.getenv('LLAMA_CPP_VISION_ENABLED', '1').strip().lower() not in {'0', 'false', 'off', 'no', ''}
    if vision_enabled:
        upload_figure_budget = settings.pdf_upload_max_figures
        if upload_figure_budget == 0:
            # Keep uploads analyzable even if the env budget is accidentally set to 0.
            upload_figure_budget = -1
        command.extend(['--max-figures', str(upload_figure_budget)])
        command.append('--describe-pages')
        command.append('--force-page-analysis')
    else:
        command.extend(['--max-figures', '0'])

    result = subprocess.run(
        command,
        cwd=settings.project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    extracted_path = Path(f'{pdf_path}_extracted.txt')
    if result.returncode != 0 or not extracted_path.exists():
        detail = (result.stderr or result.stdout or 'Unknown PDF conversion failure.').strip()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'PDF conversion failed. {detail}',
        )
    return extracted_path.resolve()


@router.get('/documents', response_model=list[DocumentResponse])
def list_documents(
    session: AppSessionRecord = Depends(get_session_or_404),
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[DocumentResponse]:
    config = default_config(get_settings().project_root)
    records = db.scalars(
        select(DocumentRecord)
        .where(DocumentRecord.user_id == current_user.id)
        .order_by(DocumentRecord.uploaded_at.desc())
    ).all()
    active_document_ids = _active_document_ids_for_session(session, db)
    return [_document_response(record, config, active_document_ids) for record in records]


@router.post('/documents/upload', response_model=UploadDocumentsResponse, status_code=201)
async def upload_documents(
    session: AppSessionRecord = Depends(get_session_or_404),
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db),
    files: list[UploadFile] = File(...),
) -> UploadDocumentsResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No files were uploaded.')

    settings = get_settings()
    library_session = get_or_create_library_session(db, current_user)
    upload_root = settings.uploads_dir / library_session.id
    upload_root.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    original_names_by_path: dict[Path, str] = {}

    for file in files:
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in {'.txt', '.pdf'}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Only .txt and .pdf files are supported.')
        original_name = Path(file.filename or 'document.txt').name
        target_path = upload_root / f'{uuid4().hex}_{original_name}'
        with target_path.open('wb') as handle:
            shutil.copyfileobj(file.file, handle)
        if suffix == '.pdf':
            extracted_path = _convert_pdf_to_text(target_path)
            saved_paths.append(extracted_path)
            original_names_by_path[extracted_path] = f'{original_name}_extracted.txt'
        else:
            resolved = target_path.resolve()
            saved_paths.append(resolved)
            original_names_by_path[resolved] = original_name

    sources = load_sources(saved_paths)
    created: list[DocumentRecord] = []
    for source in sources:
        original_name = original_names_by_path.get(source.path, source.name)
        record = db.scalar(
            select(DocumentRecord).where(
                DocumentRecord.user_id == current_user.id,
                DocumentRecord.original_name == original_name,
                DocumentRecord.sha256 == source.sha256,
            )
        )
        if record is None:
            record = DocumentRecord(
                id=uuid4().hex,
                session_id=library_session.id,
                user_id=current_user.id,
                original_name=original_name,
                storage_path=str(source.path),
                content_text=source.content,
                sha256=source.sha256,
                size_bytes=source.size_bytes,
                modified_at=_modified_at_from_source(source.modified_at),
            )
            db.add(record)
        created.append(record)

    db.add(
        SessionEventRecord(
            id=event_id(),
            session_id=session.id,
            event_type='documents_uploaded',
            payload={'count': len(created), 'file_names': [record.original_name for record in created]},
        )
    )
    session.updated_at = utc_now()
    db.commit()

    config = default_config(get_settings().project_root)
    uploaded = db.scalars(
        select(DocumentRecord)
        .where(DocumentRecord.user_id == current_user.id)
        .order_by(DocumentRecord.uploaded_at.desc())
    ).all()
    active_document_ids = _active_document_ids_for_session(session, db)
    return UploadDocumentsResponse(
        documents=[_document_response(record, config, active_document_ids) for record in uploaded]
    )


@router.post('/corpus/activate', response_model=ActiveCorpusResponse)
def activate_corpus(
    payload: ActivateCorpusRequest,
    session: AppSessionRecord = Depends(get_session_or_404),
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ActiveCorpusResponse:
    records = db.scalars(
        select(DocumentRecord)
        .where(DocumentRecord.id.in_(payload.document_ids), DocumentRecord.user_id == current_user.id)
        .order_by(DocumentRecord.uploaded_at.asc())
    ).all()
    if not records or len(records) != len(set(payload.document_ids)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='One or more documents were not found.')

    runtime_manager = get_runtime_manager()
    try:
        active = runtime_manager.activate_sources(
            session.id,
            [source_text_from_document_record(record) for record in records],
        )
    except LlamaCppInferenceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    source_names = [record.original_name for record in records]
    dataset_label = safe_label(source_names)

    db.execute(
        update(CorpusRecord)
        .where(CorpusRecord.session_id == session.id)
        .values(is_active=False)
    )
    corpus = db.scalar(
        select(CorpusRecord).where(
            CorpusRecord.session_id == session.id,
            CorpusRecord.dataset_id == active.dataset_id,
        )
    )
    if corpus is None:
        corpus = CorpusRecord(
            id=uuid4().hex,
            session_id=session.id,
            dataset_id=active.dataset_id,
            dataset_label=dataset_label,
            vector_directory=str(active.vector_directory),
            chunk_count=active.chunk_count,
            source_names=source_names,
            source_paths=active.source_paths,
            is_active=True,
        )
        db.add(corpus)
        db.flush()
        for record in records:
            db.add(CorpusDocumentRecord(corpus_id=corpus.id, document_id=record.id))
    else:
        corpus.dataset_label = dataset_label
        corpus.vector_directory = str(active.vector_directory)
        corpus.chunk_count = active.chunk_count
        corpus.source_names = source_names
        corpus.source_paths = active.source_paths
        corpus.is_active = True

    session.active_corpus_id = corpus.id
    db.add(
        SessionEventRecord(
            id=event_id(),
            session_id=session.id,
            event_type='corpus_activated',
            payload={'dataset_id': active.dataset_id, 'document_ids': payload.document_ids},
        )
    )
    session.updated_at = utc_now()
    db.commit()
    db.refresh(corpus)
    active.source_names = source_names
    active.dataset_label = dataset_label
    return active_corpus_response(active, corpus_id=corpus.id, document_ids=payload.document_ids)


def _active_document_ids_for_session(session: AppSessionRecord, db: Session) -> set[str]:
    if not session.active_corpus_id:
        return set()
    return set(
        db.scalars(select(CorpusDocumentRecord.document_id).where(CorpusDocumentRecord.corpus_id == session.active_corpus_id)).all()
    )


def _document_response(
    record: DocumentRecord,
    config,
    active_document_ids: set[str] | None = None,
) -> DocumentResponse:
    ready, count = document_chunk_status(
        Path(record.storage_path),
        config.retrieval,
        config.storage.chunk_cache_dir,
    )
    return DocumentResponse(
        id=record.id,
        original_name=record.original_name,
        storage_path=record.storage_path,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        modified_at=record.modified_at,
        uploaded_at=record.uploaded_at,
        chunk_cache_ready=ready,
        cached_chunk_count=count,
        is_active=record.id in (active_document_ids or set()),
    )


@router.delete(
    '/documents/{document_id}',
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_document(
    document_id: str,
    session: AppSessionRecord = Depends(get_session_or_404),
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    record = db.scalar(
        select(DocumentRecord).where(DocumentRecord.id == document_id, DocumentRecord.user_id == current_user.id)
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Document not found.')

    affected_corpora = db.scalars(
        select(CorpusRecord).join(CorpusDocumentRecord, CorpusDocumentRecord.corpus_id == CorpusRecord.id).where(
            CorpusDocumentRecord.document_id == document_id
        )
    ).all()
    affected_corpus_ids = [corpus.id for corpus in affected_corpora]
    affected_session_ids = [corpus.session_id for corpus in affected_corpora]

    if affected_corpus_ids:
        generated_question_ids = db.scalars(
            select(GeneratedQuestionRecord.id).where(GeneratedQuestionRecord.corpus_id.in_(affected_corpus_ids))
        ).all()
        db.execute(
            update(AppSessionRecord)
            .where(AppSessionRecord.active_corpus_id.in_(affected_corpus_ids))
            .values(active_corpus_id=None)
        )
        if generated_question_ids:
            db.execute(
                delete(AnswerAttemptRecord).where(AnswerAttemptRecord.generated_question_id.in_(generated_question_ids))
            )
        db.execute(delete(GeneratedQuestionRecord).where(GeneratedQuestionRecord.corpus_id.in_(affected_corpus_ids)))
        db.execute(delete(CorpusDocumentRecord).where(CorpusDocumentRecord.corpus_id.in_(affected_corpus_ids)))
        db.execute(delete(CorpusRecord).where(CorpusRecord.id.in_(affected_corpus_ids)))

    db.execute(delete(DocumentRecord).where(DocumentRecord.id == document_id))
    db.add(
        SessionEventRecord(
            id=event_id(),
            session_id=session.id,
            event_type='document_deleted',
            payload={'document_id': document_id, 'original_name': record.original_name},
        )
    )
    session.updated_at = utc_now()
    db.commit()

    config = default_config(get_settings().project_root)
    clear_document_chunk_status(
        Path(record.storage_path),
        config.retrieval,
        config.storage.chunk_cache_dir,
    )

    try:
        Path(record.storage_path).unlink(missing_ok=True)
    except OSError:
        pass

    runtime_manager = get_runtime_manager()
    for affected_session_id in affected_session_ids:
        runtime_manager.reset_session(affected_session_id)


@router.get('/corpus/active', response_model=ActiveCorpusResponse)
def get_active_corpus(
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> ActiveCorpusResponse:
    if not session.active_corpus_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No active corpus is selected.')
    corpus = db.get(CorpusRecord, session.active_corpus_id)
    if corpus is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Active corpus metadata was not found.')
    document_ids = db.query(CorpusDocumentRecord.document_id).filter_by(corpus_id=corpus.id).all()
    return ActiveCorpusResponse(
        corpus_id=corpus.id,
        dataset_id=corpus.dataset_id,
        dataset_label=corpus.dataset_label,
        source_names=list(corpus.source_names or []),
        source_paths=list(corpus.source_paths or []),
        chunk_count=corpus.chunk_count,
        vector_directory=corpus.vector_directory,
        document_ids=[item[0] for item in document_ids],
    )
