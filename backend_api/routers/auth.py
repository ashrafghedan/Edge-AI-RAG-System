from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import (
    get_current_token_record,
    get_current_user,
    get_or_create_library_session,
    hash_password,
    issue_access_token,
    normalize_display_name,
    normalize_email,
    user_response,
    verify_password,
)
from ..models import AppSessionRecord, AuthTokenRecord, DocumentRecord, UserRecord
from ..runtime import get_runtime_manager
from ..schemas import AuthResponse, LoginRequest, SignUpRequest, UserResponse


router = APIRouter(prefix='/auth', tags=['auth'])


def _register_user(payload: SignUpRequest, db: Session) -> AuthResponse:
    email = normalize_email(payload.email)
    if db.scalar(select(UserRecord).where(UserRecord.email == email)) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='An account with this email already exists.')

    existing_user_count = db.scalar(select(func.count()).select_from(UserRecord)) or 0
    user = UserRecord(
        id=uuid4().hex,
        email=email,
        display_name=normalize_display_name(payload.display_name, email),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.flush()
    get_or_create_library_session(db, user)
    _adopt_legacy_data_if_needed(db, user, existing_user_count)
    token = issue_access_token(db, user)
    db.commit()
    return AuthResponse(access_token=token, user=user_response(user))


@router.post('/signup', response_model=AuthResponse, status_code=201)
def signup(payload: SignUpRequest, db: Session = Depends(get_db)) -> AuthResponse:
    return _register_user(payload, db)


@router.post('/register', response_model=AuthResponse, status_code=201)
def register(payload: SignUpRequest, db: Session = Depends(get_db)) -> AuthResponse:
    return _register_user(payload, db)


@router.post('/login', response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = normalize_email(payload.email)
    user = db.scalar(select(UserRecord).where(UserRecord.email == email))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid email or password.')
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='This account is disabled.')
    get_or_create_library_session(db, user)
    token = issue_access_token(db, user)
    db.commit()
    return AuthResponse(access_token=token, user=user_response(user))


@router.get('/me', response_model=UserResponse)
def me(current_user: UserRecord = Depends(get_current_user)) -> UserResponse:
    return user_response(current_user)


@router.post('/logout', status_code=204, response_class=Response, response_model=None)
def logout(
    token: AuthTokenRecord = Depends(get_current_token_record),
    db: Session = Depends(get_db),
) -> Response:
    db.execute(delete(AuthTokenRecord).where(AuthTokenRecord.id == token.id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _adopt_legacy_data_if_needed(db: Session, user: UserRecord, existing_user_count: int) -> None:
    if existing_user_count != 0:
        return

    db.execute(
        update(AppSessionRecord)
        .where(AppSessionRecord.user_id.is_(None))
        .values(user_id=user.id)
    )
    db.execute(
        update(DocumentRecord)
        .where(DocumentRecord.user_id.is_(None))
        .values(user_id=user.id)
    )
    library = get_or_create_library_session(db, user)
    db.execute(
        update(AppSessionRecord)
        .where(AppSessionRecord.is_library.is_(True), AppSessionRecord.user_id == user.id)
        .values(title='Document Library')
    )
    db.execute(
        update(DocumentRecord)
        .where(DocumentRecord.session_id == '__document_library__')
        .values(session_id=library.id, user_id=user.id)
    )
    db.execute(delete(AppSessionRecord).where(AppSessionRecord.id == '__document_library__'))
    for runtime_session in db.scalars(select(AppSessionRecord.id).where(AppSessionRecord.user_id == user.id)).all():
        get_runtime_manager().reset_session(runtime_session)
