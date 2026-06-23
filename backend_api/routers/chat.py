from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, AsyncIterator
from uuid import uuid4

from contextlib import contextmanager

from edge_rag.chatting import finalize_chat_answer, generate_chat_response, prepare_chat_request
from edge_rag.llm_client import (
    LlamaCppInferenceError,
    LlamaCppResourceLimitError,
    StreamChunk,
    normalize_inference_error,
)
from edge_rag.policy import get_default_security_policy
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal

from ..config import get_settings
from ..database import get_db
from ..deps import event_id, get_session_or_404
from ..models import AppSessionRecord, ChatMessageRecord, SessionEventRecord, utc_now
from ..schemas import ChatMessageResponse, ChatRequest, SpeechToTextResponse
from ..runtime import get_runtime_manager
from ..speech import SpeechToTextError, SpeechToTextUnavailableError, get_transcription_service
from .documents import _convert_pdf_to_text


router = APIRouter(prefix='/sessions/{session_id}/chat', tags=['chat'])

_TEXT_ATTACHMENT_SUFFIXES = {'.txt'}
_IMAGE_ATTACHMENT_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
_PDF_ATTACHMENT_SUFFIXES = {'.pdf'}
_ALLOWED_ATTACHMENT_SUFFIXES = _TEXT_ATTACHMENT_SUFFIXES | _IMAGE_ATTACHMENT_SUFFIXES | _PDF_ATTACHMENT_SUFFIXES
_DEFAULT_ATTACHMENT_MESSAGE = 'Please analyze the attached file(s).'
_ALLOWED_AUDIO_SUFFIXES = {'.m4a', '.mp3', '.ogg', '.wav', '.weba', '.webm'}
_INTERNAL_CHAT_PAYLOAD_KEYS = {'exclude_from_history', 'safety_refusal', 'blocked_stage', 'blocked_category'}
_CONTENT_TYPE_SUFFIXES = {
    'audio/mp4': '.m4a',
    'audio/mpeg': '.mp3',
    'audio/ogg': '.ogg',
    'audio/wav': '.wav',
    'audio/webm': '.webm',
}


@router.get('/messages', response_model=list[ChatMessageResponse])
def list_chat_messages(
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> list[ChatMessageResponse]:
    messages = db.scalars(
        select(ChatMessageRecord)
        .where(ChatMessageRecord.session_id == session.id, ChatMessageRecord.mode == 'general')
        .order_by(ChatMessageRecord.created_at.asc())
    ).all()
    return [
        _chat_message_response(record, session.id)
        for record in messages
    ]


@router.post('/messages', response_model=ChatMessageResponse)
async def create_chat_message(
    request: Request,
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
) -> ChatMessageResponse:
    settings = get_settings()
    message, uploaded_files = await _parse_submission(request)
    attachments = await _store_chat_attachments(session.id, uploaded_files) if uploaded_files else []
    trimmed_message = message.strip() or (_DEFAULT_ATTACHMENT_MESSAGE if attachments else '')
    if not trimmed_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Enter a message or attach at least one file.')

    history_records = db.scalars(
        select(ChatMessageRecord)
        .where(ChatMessageRecord.session_id == session.id, ChatMessageRecord.mode == 'general')
        .order_by(ChatMessageRecord.created_at.desc())
        .limit(settings.chat_history_limit * 2)
    ).all()
    policy = get_default_security_policy()
    history = [
        item
        for record in reversed(history_records)
        if (item := _history_message_from_record(record, policy=policy)) is not None
    ]

    user_payload = {'attachments': [_attachment_payload_item(item) for item in attachments]}
    user_message = ChatMessageRecord(
        id=uuid4().hex,
        session_id=session.id,
        mode='general',
        role='user',
        content=trimmed_message,
        payload=user_payload,
    )
    db.add(user_message)

    try:
        app = get_runtime_manager().get_app(session.id)
        if hasattr(app, 'chat_result'):
            chat_result = app.chat_result(
                trimmed_message,
                history=history,
                attachments=attachments,
            )
            answer = chat_result.answer
        else:
            chat_result = None
            answer = app.chat(
                trimmed_message,
                history=history,
                attachments=attachments,
            )
    except LlamaCppInferenceError as exc:
        db.rollback()
        _raise_inference_http_error(exc)
    assistant_payload: dict[str, Any] = {}
    if chat_result is not None and chat_result.blocked:
        assistant_payload.update(
            {
                'exclude_from_history': True,
                'safety_refusal': True,
                'blocked_stage': chat_result.blocked_stage,
                'blocked_category': chat_result.blocked_category,
            }
        )
        if chat_result.blocked_stage == 'user_query':
            user_message.payload = {
                **user_payload,
                'exclude_from_history': True,
                'blocked_stage': chat_result.blocked_stage,
                'blocked_category': chat_result.blocked_category,
            }
    assistant_message = ChatMessageRecord(
        id=uuid4().hex,
        session_id=session.id,
        mode='general',
        role='assistant',
        content=answer,
        payload=assistant_payload,
    )
    db.add(assistant_message)
    db.add(
        SessionEventRecord(
            id=event_id(),
            session_id=session.id,
            event_type='general_chat_message',
            payload={
                'user_message_id': user_message.id,
                'assistant_message_id': assistant_message.id,
                'attachment_names': [item['name'] for item in attachments],
            },
        )
    )
    if _is_default_session_title(session.title):
        title_seed = trimmed_message if trimmed_message != _DEFAULT_ATTACHMENT_MESSAGE else _title_from_attachments(attachments)
        session.title = _title_from_message(title_seed) or session.title
    session.updated_at = utc_now()
    db.commit()
    db.refresh(assistant_message)
    return _chat_message_response(assistant_message, session.id)


@router.post('/messages/stream')
async def stream_chat_message(
    request: Request,
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
):
    """Send a chat message and stream the assistant's reply as SSE.

    Emits one ``data: <json>\\n\\n`` event per chunk. Event shapes:

    * ``{"type": "analyzing"}``                  - sent immediately
    * ``{"type": "user", "message": {...}}``     - the persisted user record
    * ``{"type": "token", "text": "..."}``       - assistant text delta
    * ``{"type": "meta", "tokens": n, ...}``     - periodic stats snapshot
    * ``{"type": "done", "message": {...}, "stats": {...}}``
    * ``{"type": "error", "detail": "..."}``     - terminal error
    """

    settings = get_settings()
    message, uploaded_files = await _parse_submission(request)
    attachments = await _store_chat_attachments(session.id, uploaded_files) if uploaded_files else []
    trimmed_message = message.strip() or (_DEFAULT_ATTACHMENT_MESSAGE if attachments else '')
    if not trimmed_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Enter a message or attach at least one file.',
        )

    history_records = db.scalars(
        select(ChatMessageRecord)
        .where(ChatMessageRecord.session_id == session.id, ChatMessageRecord.mode == 'general')
        .order_by(ChatMessageRecord.created_at.desc())
        .limit(settings.chat_history_limit * 2)
    ).all()
    policy = get_default_security_policy()
    history = [
        item
        for record in reversed(history_records)
        if (item := _history_message_from_record(record, policy=policy)) is not None
    ]

    user_payload = {'attachments': [_attachment_payload_item(item) for item in attachments]}
    user_message = ChatMessageRecord(
        id=uuid4().hex,
        session_id=session.id,
        mode='general',
        role='user',
        content=trimmed_message,
        payload=user_payload,
    )
    db.add(user_message)
    if _is_default_session_title(session.title):
        title_seed = (
            trimmed_message if trimmed_message != _DEFAULT_ATTACHMENT_MESSAGE else _title_from_attachments(attachments)
        )
        session.title = _title_from_message(title_seed) or session.title
    session.updated_at = utc_now()
    db.commit()
    db.refresh(user_message)

    user_record_payload = _chat_message_response(user_message, session.id).model_dump(mode='json')

    runtime_manager = get_runtime_manager()
    app_instance = runtime_manager.get_app(session.id)
    clients = app_instance.clients

    prep = prepare_chat_request(
        trimmed_message,
        history=history,
        attachments=attachments,
        security_policy=policy,
    )

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=128)
    stream_finished = asyncio.Event()
    final_state: dict[str, Any] = {'answer': '', 'stats': {}, 'error': None}

    def _put(event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop telemetry rather than back-pressuring the model loop.
            pass

    def _producer() -> None:
        try:
            _put({'type': 'analyzing'})
            _put({'type': 'user', 'message': user_record_payload})

            if prep.blocked_response is not None:
                final_state['answer'] = prep.blocked_response.answer
                final_state['stats'] = {
                    'blocked': True,
                    'blocked_stage': prep.blocked_response.blocked_stage,
                    'blocked_category': prep.blocked_response.blocked_category,
                }
                # Emit the refusal text up-front so the UI can render it.
                _put({'type': 'token', 'text': prep.blocked_response.answer})
                return

            accumulated = ''
            reasoning_text = ''
            stats: dict[str, Any] = {}
            try:
                for chunk in clients.stream_chat_completion(
                    prep.messages,
                    use_vision_model=prep.requires_vision,
                ):
                    if not isinstance(chunk, StreamChunk):
                        continue
                    if chunk.kind == 'token':
                        accumulated += chunk.text
                        _put({'type': 'token', 'text': chunk.text})
                    elif chunk.kind == 'reasoning':
                        reasoning_text += chunk.text
                        _put({'type': 'reasoning_token', 'text': chunk.text})
                    elif chunk.kind == 'meta' and chunk.payload:
                        stats.update(chunk.payload)
                        _put({'type': 'meta', **chunk.payload})
                    elif chunk.kind == 'done' and chunk.payload:
                        stats.update(chunk.payload)
                        accumulated = chunk.payload.get('answer') or accumulated
                        if chunk.payload.get('reasoning'):
                            reasoning_text = chunk.payload['reasoning']
                    elif chunk.kind == 'error':
                        final_state['error'] = chunk.text or 'Streaming failed.'
                        return
            except Exception as exc:
                normalized = normalize_inference_error(
                    exc, base_url=clients.config.ollama_base_url
                )
                final_state['error'] = str(normalized) if normalized else str(exc)
                return

            if not accumulated.strip() and reasoning_text.strip():
                # If the model still failed to produce a visible answer after a
                # focused retry, show a plain user-facing fallback instead of
                # leaking internal reasoning/budget implementation details.
                accumulated = (
                    'I could not finish the answer this time. Please try again.'
                )
                _put({'type': 'token', 'text': accumulated})

            final = finalize_chat_answer(
                trimmed_message,
                accumulated,
                prep.selected_history,
                security_policy=policy,
            )
            final_state['answer'] = final.answer
            final_state['reasoning'] = reasoning_text
            final_state['stats'] = {
                **stats,
                'blocked': final.blocked,
                'blocked_stage': final.blocked_stage,
                'blocked_category': final.blocked_category,
            }
            if final.answer != accumulated:
                # Policy rewrote the answer (e.g. refusal). Send the difference so
                # the rendered text matches what we persist.
                _put({'type': 'replace', 'text': final.answer})
        finally:
            queue.put_nowait(None)
            stream_finished.set()

    producer_task = asyncio.get_running_loop().run_in_executor(None, _producer)

    async def _event_stream() -> AsyncIterator[bytes]:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f'data: {json.dumps(item, ensure_ascii=False)}\n\n'.encode('utf-8')

            await producer_task

            if final_state.get('error'):
                error_event = {'type': 'error', 'detail': final_state['error']}
                # Roll back the optimistic user message so the user can retry.
                with _scoped_session(session.id) as session_db:
                    record = session_db.get(ChatMessageRecord, user_message.id)
                    if record is not None:
                        session_db.delete(record)
                yield f'data: {json.dumps(error_event, ensure_ascii=False)}\n\n'.encode('utf-8')
                return

            assistant_payload: dict[str, Any] = {}
            stats = final_state.get('stats') or {}
            if final_state.get('reasoning'):
                assistant_payload['reasoning'] = final_state['reasoning']
            if stats.get('blocked'):
                assistant_payload.update(
                    {
                        'exclude_from_history': True,
                        'safety_refusal': True,
                        'blocked_stage': stats.get('blocked_stage'),
                        'blocked_category': stats.get('blocked_category'),
                    }
                )

            with _scoped_session(session.id) as session_db:
                assistant_message = ChatMessageRecord(
                    id=uuid4().hex,
                    session_id=session.id,
                    mode='general',
                    role='assistant',
                    content=final_state.get('answer') or '',
                    payload=assistant_payload,
                )
                session_db.add(assistant_message)
                session_db.add(
                    SessionEventRecord(
                        id=event_id(),
                        session_id=session.id,
                        event_type='general_chat_message',
                        payload={
                            'user_message_id': user_message.id,
                            'assistant_message_id': assistant_message.id,
                            'attachment_names': [item['name'] for item in attachments],
                            'streamed': True,
                            'tokens': stats.get('tokens'),
                            'elapsed_seconds': stats.get('elapsed_seconds'),
                        },
                    )
                )
                refreshed_session = session_db.get(AppSessionRecord, session.id)
                if refreshed_session is not None:
                    refreshed_session.updated_at = utc_now()
                session_db.commit()
                session_db.refresh(assistant_message)
                assistant_payload_response = _chat_message_response(assistant_message, session.id).model_dump(mode='json')

            done_event = {
                'type': 'done',
                'message': assistant_payload_response,
                'stats': {
                    key: value
                    for key, value in (final_state.get('stats') or {}).items()
                    if key
                    in {
                        'tokens',
                        'reasoning_tokens',
                        'prompt_tokens',
                        'elapsed_seconds',
                        'tokens_per_second',
                        'continuations',
                    }
                },
                'reasoning': final_state.get('reasoning') or '',
            }
            yield f'data: {json.dumps(done_event, ensure_ascii=False)}\n\n'.encode('utf-8')
        finally:
            stream_finished.set()

    return StreamingResponse(
        _event_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


@router.post('/transcribe', response_model=SpeechToTextResponse)
async def transcribe_chat_audio(
    language: str | None = Form(default=None),
    file: UploadFile = File(...),
    session: AppSessionRecord = Depends(get_session_or_404),
) -> SpeechToTextResponse:
    settings = get_settings()
    payload = await file.read()
    try:
        if not payload:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Upload a non-empty audio file.')

        if len(payload) > settings.stt_max_file_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f'Audio uploads are limited to {settings.stt_max_file_mb} MB.',
            )

        suffix = _audio_suffix_for_upload(file)
        if suffix not in _ALLOWED_AUDIO_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Speech uploads support .webm, .weba, .wav, .ogg, .mp3, and .m4a audio files.',
            )

        upload_root = settings.uploads_dir / session.id / 'speech'
        upload_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=upload_root, suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)

        try:
            result = get_transcription_service().transcribe_file(temp_path, language=language)
        finally:
            temp_path.unlink(missing_ok=True)
    except SpeechToTextUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except SpeechToTextError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    finally:
        await file.close()

    return SpeechToTextResponse(text=result.text, language=result.language)


@router.get('/messages/{message_id}/attachments/{attachment_index}')
def open_chat_attachment(
    message_id: str,
    attachment_index: int,
    session: AppSessionRecord = Depends(get_session_or_404),
    db: Session = Depends(get_db),
):
    record = db.scalar(
        select(ChatMessageRecord).where(
            ChatMessageRecord.id == message_id,
            ChatMessageRecord.session_id == session.id,
            ChatMessageRecord.mode == 'general',
        )
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Message not found.')

    attachments = (record.payload or {}).get('attachments') or []
    if attachment_index < 0 or attachment_index >= len(attachments):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Attachment not found.')

    attachment = attachments[attachment_index]
    path = Path(str(attachment.get('storage_path') or '')).resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Attachment file is missing.')

    media_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return FileResponse(path, media_type=media_type, filename=str(attachment.get('name') or path.name))


async def _parse_submission(request: Request) -> tuple[str, list[UploadFile]]:
    content_type = request.headers.get('content-type', '').lower()
    if 'multipart/form-data' in content_type:
        form = await request.form()
        message = str(form.get('message') or '')
        files = [
            item
            for key, item in form.multi_items()
            if key == 'files' and hasattr(item, 'filename') and item.filename
        ]
        return message, files

    try:
        payload = ChatRequest.model_validate(await request.json())
    except Exception as exc:  # pragma: no cover - FastAPI request parsing detail
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid chat request payload.') from exc
    return payload.message, []


async def _store_chat_attachments(session_id: str, files: list[UploadFile]) -> list[dict[str, Any]]:
    settings = get_settings()
    upload_root = settings.uploads_dir / session_id / 'chat'
    upload_root.mkdir(parents=True, exist_ok=True)
    attachments: list[dict[str, Any]] = []

    for file in files:
        try:
            original_name = Path(file.filename or 'attachment').name
            suffix = Path(original_name).suffix.lower()
            if suffix not in _ALLOWED_ATTACHMENT_SUFFIXES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Assistant uploads support .txt, .pdf, and common image files.',
                )

            target_path = upload_root / f'{uuid4().hex}_{original_name}'
            with target_path.open('wb') as handle:
                shutil.copyfileobj(file.file, handle)

            if suffix in _TEXT_ATTACHMENT_SUFFIXES:
                text_content = target_path.read_text(encoding='utf-8', errors='ignore').strip()
                attachments.append(
                    {
                        'name': original_name,
                        'kind': 'text',
                        'storage_path': str(target_path.resolve()),
                        'text_path': str(target_path.resolve()),
                        'text_content': text_content,
                        'text_excerpt': _truncate_text(text_content),
                    }
                )
            elif suffix in _PDF_ATTACHMENT_SUFFIXES:
                text_path = _convert_pdf_to_text(target_path)
                text_content = text_path.read_text(encoding='utf-8', errors='ignore').strip()
                attachments.append(
                    {
                        'name': original_name,
                        'kind': 'pdf',
                        'storage_path': str(target_path.resolve()),
                        'text_path': str(text_path.resolve()),
                        'text_content': text_content,
                        'text_excerpt': _truncate_text(text_content),
                    }
                )
            else:
                attachments.append(
                    {
                        'name': original_name,
                        'kind': 'image',
                        'storage_path': str(target_path.resolve()),
                    }
                )
        finally:
            await file.close()

    return attachments


def _attachment_payload_item(attachment: dict[str, Any]) -> dict[str, Any]:
    payload = {
        'name': attachment['name'],
        'kind': attachment['kind'],
        'storage_path': attachment['storage_path'],
    }
    if attachment.get('text_path'):
        payload['text_path'] = attachment['text_path']
    if attachment.get('text_excerpt'):
        payload['text_excerpt'] = attachment['text_excerpt']
    return payload


def _history_message_from_record(record: ChatMessageRecord, *, policy) -> dict[str, Any] | None:
    payload = dict(record.payload or {})
    if payload.get('exclude_from_history'):
        return None
    if record.role == 'assistant' and (payload.get('safety_refusal') or policy.is_refusal_text(record.content)):
        return None
    if record.role == 'assistant' and policy.evaluate_text(record.content, purpose='model_output').blocked:
        return None
    if record.role == 'user' and payload.get('blocked_stage') == 'user_query':
        return None
    if record.role == 'user' and policy.evaluate_text(record.content, purpose='user_query').blocked:
        return None
    public_payload = {key: value for key, value in payload.items() if key not in _INTERNAL_CHAT_PAYLOAD_KEYS}
    return {'role': record.role, 'content': record.content, 'payload': public_payload}


def _chat_message_response(record: ChatMessageRecord, session_id: str) -> ChatMessageResponse:
    payload = {
        key: value
        for key, value in dict(record.payload or {}).items()
        if key not in _INTERNAL_CHAT_PAYLOAD_KEYS
    }
    attachments = payload.get('attachments') or []
    if attachments:
        payload['attachments'] = [
            {
                **item,
                'download_url': f"/api/v1/sessions/{session_id}/chat/messages/{record.id}/attachments/{index}",
            }
            for index, item in enumerate(attachments)
        ]
    return ChatMessageResponse(
        id=record.id,
        mode=record.mode,
        role=record.role,
        content=record.content,
        payload=payload,
        created_at=record.created_at,
    )


def _truncate_text(value: str, limit: int = 4000) -> str:
    compact = value.strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + '...'


def _title_from_attachments(attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return ''
    if len(attachments) == 1:
        attachment = attachments[0]
        kind = str(attachment.get('kind') or 'file').strip().title()
        return f'{kind}: {attachment["name"]}'
    return ', '.join(item['name'] for item in attachments[:2])


def _is_default_session_title(title: str) -> bool:
    normalized = ' '.join(str(title or '').split()).lower()
    return (
        normalized == 'new chat'
        or normalized.startswith('new chat ')
        or normalized.startswith('learning session ')
    )


def _title_from_message(message: str, limit: int = 60) -> str:
    compact = ' '.join(str(message or '').split())
    if not compact:
        return ''

    compact = re.sub(r'\s+', ' ', compact).strip()
    words = compact.split()
    title = ' '.join(words[:10]) if len(words) > 10 else compact
    if len(title) > limit:
        title = title[: limit - 3].rstrip() + '...'
    return title


def _raise_inference_http_error(exc: LlamaCppInferenceError) -> None:
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if isinstance(exc, LlamaCppResourceLimitError)
        else status.HTTP_502_BAD_GATEWAY
    )
    raise HTTPException(status_code=status_code, detail=str(exc)) from exc


def _audio_suffix_for_upload(file: UploadFile) -> str:
    name_suffix = Path(file.filename or '').suffix.lower()
    if name_suffix:
        return name_suffix
    content_type = (file.content_type or '').split(';', maxsplit=1)[0].strip().lower()
    return _CONTENT_TYPE_SUFFIXES.get(content_type, '')


@contextmanager
def _scoped_session(_session_id: str):
    """Yield a short-lived SQLAlchemy session for use inside the SSE generator."""

    session_db = SessionLocal()
    try:
        yield session_db
    finally:
        session_db.close()
