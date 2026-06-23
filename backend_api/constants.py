LIBRARY_SESSION_ID = '__document_library__'
LIBRARY_SESSION_PREFIX = '__document_library__:'
LIBRARY_SESSION_TITLE = 'Document Library'
AUTH_TOKEN_TTL_DAYS = 30


def library_session_id_for_user(user_id: str) -> str:
    return f'{LIBRARY_SESSION_PREFIX}{user_id}'


def is_library_session_id(session_id: str) -> bool:
    return session_id == LIBRARY_SESSION_ID or session_id.startswith(LIBRARY_SESSION_PREFIX)
