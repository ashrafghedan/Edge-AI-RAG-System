from __future__ import annotations

from uuid import uuid4

from edge_rag.llm_client import LlamaCppInferenceError, LlamaCppResourceLimitError
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import (
    ensure_runtime_active_corpus,
    event_id,
    get_session_or_404,
    restore_generated_question,
)
from ..models import (
    AnswerAttemptRecord,
    AppSessionRecord,
    ChatMessageRecord,
    GeneratedQuestionRecord,
    SessionEventRecord,
    utc_now,
)
from ..runtime import get_runtime_manager
from ..schemas import (
    GeneratedQuestionResponse,
    GradeQuestionRequest,
    GradeQuestionResponse,
    GroundedAnswerResponse,
    RAGAskRequest,
)


router = APIRouter(prefix='/sessions/{session_id}/learning', tags=['learning'])


@router.post('/ask', response_model=GroundedAnswerResponse)
def ask_grounded_question(
    payload: RAGAskRequest,
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> GroundedAnswerResponse:
    runtime_manager = get_runtime_manager()
    active, corpus, _document_ids = ensure_runtime_active_corpus(session.id, db, runtime_manager)
    app = runtime_manager.get_app(session.id)
    try:
        result = app.ask_question(payload.question)
    except LlamaCppInferenceError as exc:
        db.rollback()
        _raise_inference_http_error(exc)

    db.add(
        ChatMessageRecord(
            id=uuid4().hex,
            session_id=session.id,
            mode='rag',
            role='user',
            content=payload.question,
            payload={'dataset_id': active.dataset_id},
        )
    )
    db.add(
        ChatMessageRecord(
            id=uuid4().hex,
            session_id=session.id,
            mode='rag',
            role='assistant',
            content=result.answer,
            payload={
                'found': result.found,
                'source_names': result.source_names,
                'evidence_ids': result.evidence_ids,
                'dataset_id': active.dataset_id,
            },
        )
    )
    db.add(
        SessionEventRecord(
            id=event_id(),
            session_id=session.id,
            event_type='grounded_question_asked',
            payload={'dataset_id': corpus.dataset_id, 'found': result.found},
        )
    )
    session.updated_at = utc_now()
    db.commit()
    return GroundedAnswerResponse(
        answer=result.answer,
        found=result.found,
        source_names=result.source_names,
        evidence_ids=result.evidence_ids,
    )


@router.get('/questions', response_model=list[GeneratedQuestionResponse])
def list_generated_questions(
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> list[GeneratedQuestionResponse]:
    records = db.scalars(
        select(GeneratedQuestionRecord)
        .where(GeneratedQuestionRecord.session_id == session.id)
        .order_by(GeneratedQuestionRecord.created_at.desc())
    ).all()
    return [
        GeneratedQuestionResponse(
            question_id=record.id,
            question=record.question_text,
            model_answer=record.model_answer,
            source_names=list(record.source_names or []),
            source_chunk_ids=list(record.source_chunk_ids or []),
            created_at=record.created_at,
        )
        for record in records
    ]


@router.post('/questions', response_model=GeneratedQuestionResponse)
def generate_question(
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> GeneratedQuestionResponse:
    runtime_manager = get_runtime_manager()
    _active, corpus, _document_ids = ensure_runtime_active_corpus(session.id, db, runtime_manager)
    app = runtime_manager.get_app(session.id)
    try:
        generated = app.generate_question()
    except LlamaCppInferenceError as exc:
        db.rollback()
        _raise_inference_http_error(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    merged_record = db.merge(
        GeneratedQuestionRecord(
        id=generated.question_id,
        session_id=session.id,
        corpus_id=corpus.id,
        question_text=generated.question,
        model_answer=generated.model_answer,
        source_names=generated.source_names,
        source_chunk_ids=generated.source_chunk_ids,
        )
    )
    db.add(
        SessionEventRecord(
            id=event_id(),
            session_id=session.id,
            event_type='generated_question',
            payload={'question_id': generated.question_id, 'dataset_id': corpus.dataset_id},
        )
    )
    session.updated_at = utc_now()
    db.commit()
    db.refresh(merged_record)
    return GeneratedQuestionResponse(
        question_id=merged_record.id,
        question=merged_record.question_text,
        model_answer=merged_record.model_answer,
        source_names=list(merged_record.source_names or []),
        source_chunk_ids=list(merged_record.source_chunk_ids or []),
        created_at=merged_record.created_at,
    )


@router.post('/questions/{question_id}/grade', response_model=GradeQuestionResponse)
def grade_question(
    question_id: str,
    payload: GradeQuestionRequest,
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> GradeQuestionResponse:
    runtime_manager = get_runtime_manager()
    _active, corpus, _document_ids = ensure_runtime_active_corpus(session.id, db, runtime_manager)
    app = runtime_manager.get_app(session.id)

    record = db.get(GeneratedQuestionRecord, question_id)
    if record is None or record.session_id != session.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Generated question not found.')

    runtime_manager.restore_question(session.id, restore_generated_question(record))
    try:
        grading = app.grade_generated_question(question_id, payload.user_answer)
    except LlamaCppInferenceError as exc:
        db.rollback()
        _raise_inference_http_error(exc)

    attempt = AnswerAttemptRecord(
        id=uuid4().hex,
        session_id=session.id,
        generated_question_id=question_id,
        user_answer=payload.user_answer,
        score=grading.score,
        feedback=grading.feedback,
        model_answer=grading.model_answer,
    )
    db.add(attempt)
    db.add(
        SessionEventRecord(
            id=event_id(),
            session_id=session.id,
            event_type='graded_answer',
            payload={'question_id': question_id, 'score': grading.score, 'dataset_id': corpus.dataset_id},
        )
    )
    session.updated_at = utc_now()
    db.commit()
    db.refresh(attempt)
    return GradeQuestionResponse(
        attempt_id=attempt.id,
        question_id=question_id,
        score=attempt.score,
        feedback=attempt.feedback,
        model_answer=attempt.model_answer,
        created_at=attempt.created_at,
    )


@router.get('/attempts', response_model=list[GradeQuestionResponse])
def list_attempts(
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> list[GradeQuestionResponse]:
    attempts = db.scalars(
        select(AnswerAttemptRecord)
        .where(AnswerAttemptRecord.session_id == session.id)
        .order_by(AnswerAttemptRecord.created_at.desc())
    ).all()
    return [
        GradeQuestionResponse(
            attempt_id=record.id,
            question_id=record.generated_question_id,
            score=record.score,
            feedback=record.feedback,
            model_answer=record.model_answer,
            created_at=record.created_at,
        )
        for record in attempts
    ]


def _raise_inference_http_error(exc: LlamaCppInferenceError) -> None:
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if isinstance(exc, LlamaCppResourceLimitError)
        else status.HTTP_502_BAD_GATEWAY
    )
    raise HTTPException(status_code=status_code, detail=str(exc)) from exc
