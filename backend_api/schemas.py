from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    email: str
    display_name: str
    initials: str
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    user: UserResponse


class SignUpRequest(BaseModel):
    email: str
    password: str
    display_name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_name: str
    storage_path: str
    sha256: str
    size_bytes: int
    modified_at: datetime
    uploaded_at: datetime
    chunk_cache_ready: bool = False
    cached_chunk_count: int | None = None
    is_active: bool = False


class ActiveCorpusResponse(BaseModel):
    corpus_id: str | None = None
    dataset_id: str
    dataset_label: str
    source_names: list[str]
    source_paths: list[str]
    chunk_count: int
    vector_directory: str
    document_ids: list[str] = Field(default_factory=list)


class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    active_corpus: ActiveCorpusResponse | None = None


class SessionListItem(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    active_corpus_label: str | None = None


class UploadDocumentsResponse(BaseModel):
    documents: list[DocumentResponse]


class ActivateCorpusRequest(BaseModel):
    document_ids: list[str]


class ChatRequest(BaseModel):
    message: str


class ChatMessageResponse(BaseModel):
    id: str
    mode: str
    role: str
    content: str
    payload: dict = Field(default_factory=dict)
    created_at: datetime


class SpeechToTextResponse(BaseModel):
    text: str
    language: str | None = None


class RAGAskRequest(BaseModel):
    question: str


class GroundedAnswerResponse(BaseModel):
    answer: str
    found: bool
    source_names: list[str]
    evidence_ids: list[str]


class GeneratedQuestionResponse(BaseModel):
    question_id: str
    question: str
    model_answer: str
    source_names: list[str]
    source_chunk_ids: list[str]
    created_at: datetime


class GradeQuestionRequest(BaseModel):
    user_answer: str


class GradeQuestionResponse(BaseModel):
    attempt_id: str
    question_id: str
    score: int
    feedback: str
    model_answer: str
    created_at: datetime
