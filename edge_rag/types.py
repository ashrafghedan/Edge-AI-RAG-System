from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document


@dataclass(slots=True)
class SourceText:
    name: str
    path: Path
    content: str
    sha256: str
    size_bytes: int
    modified_at: str


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    source_name: str
    content: str
    distance: float
    relevance: float


@dataclass(slots=True)
class GroundedAnswerResult:
    answer: str
    found: bool
    source_names: list[str]
    evidence_ids: list[str]
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list, repr=False)


@dataclass(slots=True)
class GeneratedQuestion:
    question_id: str
    question: str
    model_answer: str
    source_names: list[str]
    source_chunk_ids: list[str]
    created_at: str


@dataclass(slots=True)
class GradingResult:
    score: int
    feedback: str
    model_answer: str


@dataclass(slots=True)
class ActiveCorpus:
    dataset_id: str
    dataset_label: str
    source_names: list[str]
    source_paths: list[str]
    vector_directory: Path
    chunk_count: int
    chunks: list[Document] = field(repr=False)


@dataclass(slots=True)
class SessionState:
    session_id: str
    active_corpus: ActiveCorpus | None = None
    generated_questions: dict[str, GeneratedQuestion] = field(default_factory=dict)
