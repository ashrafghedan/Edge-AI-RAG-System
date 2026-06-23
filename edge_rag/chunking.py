from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RetrievalConfig
from .types import SourceText
from .utils import atomic_write_json


_CHUNK_CACHE_VERSION = 1


def _strip_known_boilerplate(text: str) -> str:
    start_match = re.search(
        r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if start_match:
        text = text[start_match.end() :]

    end_match = re.search(
        r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if end_match:
        text = text[: end_match.start()]

    return text


def clean_text(text: str) -> str:
    cleaned = _strip_known_boilerplate(text)
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', ' ')
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def chunk_cache_path(source_sha256: str, config: RetrievalConfig, cache_dir: Path) -> Path:
    return cache_dir / (
        f'{source_sha256}_cs{int(config.chunk_size)}_co{int(config.chunk_overlap)}_v{_CHUNK_CACHE_VERSION}.json'
    )


def chunk_cache_status(
    source_sha256: str,
    config: RetrievalConfig,
    cache_dir: Path,
) -> tuple[bool, int | None]:
    path = chunk_cache_path(source_sha256, config, cache_dir)
    if not path.exists():
        return False, None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, dict):
        return False, None
    if payload.get('version') != _CHUNK_CACHE_VERSION:
        return False, None
    if payload.get('source_sha256') != source_sha256:
        return False, None
    count = payload.get('chunk_count')
    if not isinstance(count, int):
        chunks = payload.get('chunks')
        count = len(chunks) if isinstance(chunks, list) else None
    return True, count


def document_chunk_status(
    source_path: Path,
    config: RetrievalConfig,
    cache_dir: Path,
) -> tuple[bool, int | None]:
    path = _document_status_path(source_path, config, cache_dir)
    if not path.exists():
        return False, None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, dict):
        return False, None
    if payload.get('version') != _CHUNK_CACHE_VERSION:
        return False, None
    count = payload.get('chunk_count')
    return True, int(count) if isinstance(count, int) else None


def mark_document_chunk_ready(
    source_path: Path,
    config: RetrievalConfig,
    cache_dir: Path,
    *,
    chunk_count: int,
    source_sha256: str,
) -> None:
    atomic_write_json(
        _document_status_path(source_path, config, cache_dir),
        {
            'version': _CHUNK_CACHE_VERSION,
            'source_path': str(Path(source_path).resolve()),
            'source_sha256': source_sha256,
            'chunk_size': int(config.chunk_size),
            'chunk_overlap': int(config.chunk_overlap),
            'chunk_count': int(chunk_count),
        },
    )


def clear_document_chunk_status(source_path: Path, config: RetrievalConfig, cache_dir: Path) -> None:
    try:
        _document_status_path(source_path, config, cache_dir).unlink()
    except FileNotFoundError:
        return


def build_chunked_documents(
    sources: list[SourceText],
    config: RetrievalConfig,
    *,
    cache_dir: Path | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=['\n\n', '\n', '. ', '? ', '! ', '; ', ': ', ' '],
    )
    documents: list[Document] = []
    for source in sources:
        cached = _load_cached_source_chunks(source, config, cache_dir)
        if cached is not None:
            documents.extend(cached)
            continue

        chunk_payloads = _chunk_payloads_for_source(source, splitter)
        if not chunk_payloads:
            continue
        source_documents = _documents_from_chunk_payloads(source, chunk_payloads)
        documents.extend(source_documents)
        _save_cached_source_chunks(source, config, chunk_payloads, cache_dir)
    if not documents:
        raise ValueError('No usable text content was found after preprocessing.')
    return documents


def _chunk_payloads_for_source(
    source: SourceText,
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict[str, int | str]]:
    cleaned = clean_text(source.content)
    if not cleaned:
        return []
    chunk_payloads: list[dict[str, int | str]] = []
    for index, chunk in enumerate(splitter.split_text(cleaned)):
        content = chunk.strip()
        if len(content) < 40:
            continue
        chunk_payloads.append({'chunk_index': index, 'page_content': content})
    return chunk_payloads


def _documents_from_chunk_payloads(
    source: SourceText,
    chunk_payloads: list[dict[str, int | str]],
) -> list[Document]:
    documents: list[Document] = []
    for item in chunk_payloads:
        chunk_index = int(item['chunk_index'])
        documents.append(
            Document(
                page_content=str(item['page_content']),
                metadata={
                    'source_name': source.name,
                    'source_path': str(source.path),
                    'source_sha256': source.sha256,
                    'chunk_index': chunk_index,
                    'chunk_id': f'{source.sha256[:12]}-{chunk_index:04d}',
                },
            )
        )
    return documents


def _load_cached_source_chunks(
    source: SourceText,
    config: RetrievalConfig,
    cache_dir: Path | None,
) -> list[Document] | None:
    if cache_dir is None:
        return None
    path = chunk_cache_path(source.sha256, config, cache_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('version') != _CHUNK_CACHE_VERSION:
        return None
    if payload.get('source_sha256') != source.sha256:
        return None
    chunk_payloads = payload.get('chunks')
    if not isinstance(chunk_payloads, list):
        return None
    normalized_payloads: list[dict[str, int | str]] = []
    for item in chunk_payloads:
        if not isinstance(item, dict):
            return None
        page_content = item.get('page_content')
        chunk_index = item.get('chunk_index')
        if not isinstance(page_content, str) or not isinstance(chunk_index, int):
            return None
        normalized_payloads.append({'page_content': page_content, 'chunk_index': chunk_index})
    mark_document_chunk_ready(
        source.path,
        config,
        cache_dir,
        chunk_count=len(normalized_payloads),
        source_sha256=source.sha256,
    )
    return _documents_from_chunk_payloads(source, normalized_payloads)


def _save_cached_source_chunks(
    source: SourceText,
    config: RetrievalConfig,
    chunk_payloads: list[dict[str, int | str]],
    cache_dir: Path | None,
) -> None:
    if cache_dir is None:
        return
    path = chunk_cache_path(source.sha256, config, cache_dir)
    atomic_write_json(
        path,
        {
            'version': _CHUNK_CACHE_VERSION,
            'source_sha256': source.sha256,
            'chunk_size': int(config.chunk_size),
            'chunk_overlap': int(config.chunk_overlap),
            'chunk_count': len(chunk_payloads),
            'chunks': chunk_payloads,
        },
    )
    mark_document_chunk_ready(
        source.path,
        config,
        cache_dir,
        chunk_count=len(chunk_payloads),
        source_sha256=source.sha256,
    )


def _document_status_path(source_path: Path, config: RetrievalConfig, cache_dir: Path) -> Path:
    digest = hashlib.sha256(str(Path(source_path).resolve()).encode('utf-8')).hexdigest()[:24]
    return cache_dir / '_documents' / (
        f'{digest}_cs{int(config.chunk_size)}_co{int(config.chunk_overlap)}_v{_CHUNK_CACHE_VERSION}.json'
    )
