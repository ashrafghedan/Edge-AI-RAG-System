from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import corpus_record_response, get_current_user, get_session_or_404
from ..models import (
    AnswerAttemptRecord,
    AppSessionRecord,
    ChatMessageRecord,
    CorpusDocumentRecord,
    CorpusRecord,
    GeneratedQuestionRecord,
    SessionEventRecord,
    UserRecord,
)
from ..runtime import get_runtime_manager
from ..schemas import SessionListItem, SessionResponse


router = APIRouter(prefix='/sessions', tags=['sessions'])


@router.get('', response_model=list[SessionListItem])
def list_sessions(
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SessionListItem]:
    records = db.scalars(
        select(AppSessionRecord)
        .where(
            AppSessionRecord.user_id == current_user.id,
            AppSessionRecord.is_library.is_(False),
        )
        .order_by(AppSessionRecord.updated_at.desc(), AppSessionRecord.created_at.desc())
        .limit(20)
    ).all()
    active_corpus_ids = [record.active_corpus_id for record in records if record.active_corpus_id]
    corpora = {
        corpus.id: corpus
        for corpus in db.scalars(select(CorpusRecord).where(CorpusRecord.id.in_(active_corpus_ids))).all()
    }
    return [
        SessionListItem(
            session_id=record.id,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
            active_corpus_label=corpora.get(record.active_corpus_id).dataset_label if record.active_corpus_id in corpora else None,
        )
        for record in records
    ]


@router.post('', response_model=SessionResponse, status_code=201)
def create_session(
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionResponse:
    session_id = uuid4().hex
    record = AppSessionRecord(
        id=session_id,
        user_id=current_user.id,
        title='New Chat',
        is_library=False,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    get_runtime_manager().get_app(session_id)
    return SessionResponse(
        session_id=record.id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
        active_corpus=None,
    )


@router.get('/{session_id}', response_model=SessionResponse)
def get_session(
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> SessionResponse:
    active_corpus = None
    if session.active_corpus_id:
        corpus = db.get(CorpusRecord, session.active_corpus_id)
        if corpus is not None:
            document_ids = db.query(CorpusDocumentRecord.document_id).filter_by(corpus_id=corpus.id).all()
            active_corpus = corpus_record_response(corpus, [item[0] for item in document_ids])
    return SessionResponse(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        active_corpus=active_corpus,
    )


@router.delete('/{session_id}', status_code=status.HTTP_204_NO_CONTENT, response_class=Response, response_model=None)
def delete_session(
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> Response:
    if session.is_library:
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    corpus_ids = db.scalars(
        select(CorpusRecord.id).where(CorpusRecord.session_id == session.id)
    ).all()

    db.execute(
        update(AppSessionRecord)
        .where(AppSessionRecord.id == session.id)
        .values(active_corpus_id=None)
    )
    db.execute(delete(AnswerAttemptRecord).where(AnswerAttemptRecord.session_id == session.id))
    db.execute(delete(ChatMessageRecord).where(ChatMessageRecord.session_id == session.id))
    db.execute(delete(SessionEventRecord).where(SessionEventRecord.session_id == session.id))
    db.execute(delete(GeneratedQuestionRecord).where(GeneratedQuestionRecord.session_id == session.id))
    if corpus_ids:
        db.execute(delete(CorpusDocumentRecord).where(CorpusDocumentRecord.corpus_id.in_(corpus_ids)))
    db.execute(delete(CorpusRecord).where(CorpusRecord.session_id == session.id))
    db.execute(delete(AppSessionRecord).where(AppSessionRecord.id == session.id))
    db.commit()

    get_runtime_manager().destroy_session(session.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
