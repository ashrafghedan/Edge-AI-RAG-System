from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env import load_project_env


@dataclass(frozen=True, slots=True)
class ModelConfig:
    answer_model: str
    embedding_model: str
    ollama_base_url: str
    embedding_base_url: str
    answer_temperature: float
    question_temperature: float
    grading_temperature: float
    num_ctx: int
    max_answer_tokens: int
    chat_continuations: int
    max_grading_tokens: int
    keep_alive: str
    num_thread: int | None = None
    vision_model: str | None = None
    request_timeout_seconds: int = 120
    vision_num_ctx: int = 2048
    vision_max_answer_tokens: int = 768
    vision_chat_continuations: int = 1
    stream_enabled: bool = True
    vision_enabled: bool = True
    mmproj_path: str | None = None
    reasoning_mode: str = 'auto'
    embedding_batch_size: int = 24


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_question_generation_attempts: int
    min_chunk_characters_for_question: int


@dataclass(frozen=True, slots=True)
class StorageConfig:
    base_dir: Path
    vector_store_dir: Path
    chunk_cache_dir: Path
    state_dir: Path
    logs_dir: Path
    interaction_log_path: Path
    question_history_path: Path
    active_selection_path: Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    models: ModelConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    not_available_response: str
    security_refusal_response: str


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_first(*names: str, default: str = '') -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != '':
            return value
    return default


def _default_num_thread() -> int:
    cpu_count = os.cpu_count() or 4
    cpu_max_percent = max(1, min(100, _env_int('EDGE_RAG_CPU_MAX_PERCENT', 80)))
    thread_budget = max(1, int(cpu_count * cpu_max_percent / 100))
    if cpu_max_percent < 100 and cpu_count > 1:
        thread_budget = min(thread_budget, cpu_count - 1)
    return max(1, thread_budget)


def default_config(project_root: Path | None = None) -> AppConfig:
    root = project_root or Path(__file__).resolve().parent.parent
    load_project_env(root)
    data_dir = root / 'data'
    storage = StorageConfig(
        base_dir=data_dir,
        vector_store_dir=data_dir / 'chroma',
        chunk_cache_dir=data_dir / 'chunk_cache',
        state_dir=data_dir / 'state',
        logs_dir=data_dir / 'logs',
        interaction_log_path=data_dir / 'logs' / 'interaction_log.json',
        question_history_path=data_dir / 'state' / 'question_history.json',
        active_selection_path=data_dir / 'state' / 'active_selection.json',
    )
    base_url = _env_first('LLAMA_CPP_BASE_URL', 'OLLAMA_BASE_URL', default='http://127.0.0.1:11436').rstrip('/')
    embedding_base = _env_first('LLAMA_CPP_EMBEDDING_BASE_URL', default=base_url).rstrip('/')
    answer_model = _env_first('LLAMA_CPP_MODEL', 'EDGE_RAG_ANSWER_MODEL', default='gemma-4-e2b-q4km')
    embedding_model = _env_first(
        'LLAMA_CPP_EMBEDDING_MODEL',
        'EDGE_RAG_EMBEDDING_MODEL',
        default=answer_model,
    )
    models = ModelConfig(
        answer_model=answer_model,
        embedding_model=embedding_model,
        ollama_base_url=base_url,
        embedding_base_url=embedding_base,
        answer_temperature=_env_float('EDGE_RAG_ANSWER_TEMPERATURE', 0.0),
        question_temperature=_env_float('EDGE_RAG_QUESTION_TEMPERATURE', 0.55),
        grading_temperature=_env_float('EDGE_RAG_GRADING_TEMPERATURE', 0.0),
        num_ctx=_env_int('LLAMA_CPP_NUM_CTX', _env_int('EDGE_RAG_NUM_CTX', 6144)),
        max_answer_tokens=_env_int(
            'LLAMA_CPP_MAX_TOKENS',
            _env_int('EDGE_RAG_MAX_ANSWER_TOKENS', 3072),
        ),
        chat_continuations=_env_int('EDGE_RAG_CHAT_CONTINUATIONS', 8),
        max_grading_tokens=_env_int('EDGE_RAG_MAX_GRADING_TOKENS', 380),
        keep_alive=os.getenv('EDGE_RAG_KEEP_ALIVE', '30s'),
        num_thread=_env_int('EDGE_RAG_NUM_THREAD', _default_num_thread()),
        vision_model=os.getenv('EDGE_RAG_VISION_MODEL') or None,
        request_timeout_seconds=_env_int(
            'LLAMA_CPP_REQUEST_TIMEOUT',
            _env_int('EDGE_RAG_OLLAMA_REQUEST_TIMEOUT', 300),
        ),
        vision_num_ctx=_env_int('EDGE_RAG_VISION_NUM_CTX', 3072),
        vision_max_answer_tokens=_env_int('EDGE_RAG_VISION_MAX_ANSWER_TOKENS', 1280),
        vision_chat_continuations=_env_int('EDGE_RAG_VISION_CHAT_CONTINUATIONS', 2),
        stream_enabled=_env_bool('LLAMA_CPP_STREAM', True),
        vision_enabled=_env_bool('LLAMA_CPP_VISION_ENABLED', True),
        mmproj_path=os.getenv('LLAMA_CPP_MMPROJ_PATH') or None,
        reasoning_mode=str(os.getenv('LLAMA_CPP_REASONING', 'auto') or 'auto').strip().lower() or 'auto',
        embedding_batch_size=max(1, _env_int('EDGE_RAG_EMBEDDING_BATCH_SIZE', 24)),
    )
    retrieval = RetrievalConfig(
        chunk_size=_env_int('EDGE_RAG_CHUNK_SIZE', 1100),
        chunk_overlap=_env_int('EDGE_RAG_CHUNK_OVERLAP', 220),
        top_k=_env_int('EDGE_RAG_TOP_K', 4),
        max_question_generation_attempts=_env_int('EDGE_RAG_QUESTION_ATTEMPTS', 8),
        min_chunk_characters_for_question=_env_int('EDGE_RAG_MIN_QUESTION_CHARS', 180),
    )
    return AppConfig(
        models=models,
        retrieval=retrieval,
        storage=storage,
        not_available_response='The information is not available in the provided documents.',
        security_refusal_response='This request is blocked by the local security policy.',
    )
