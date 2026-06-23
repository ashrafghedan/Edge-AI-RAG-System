from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import suppress
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import AppConfig
from .loaders import build_dataset_id, source_identity_sort_key
from .types import SourceText
from .utils import atomic_write_json


_INDEX_FORMAT_VERSION = 3
_EMBEDDING_CACHE_VERSION = 1
_DATASET_LOCKS: dict[str, RLock] = {}
_DATASET_LOCKS_GUARD = Lock()


def _dataset_lock(dataset_dir: Path) -> RLock:
    key = str(dataset_dir.resolve())
    with _DATASET_LOCKS_GUARD:
        lock = _DATASET_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _DATASET_LOCKS[key] = lock
        return lock


class ChromaIndexManager:
    def __init__(self, config: AppConfig, embeddings: Any) -> None:
        self.config = config
        self.embeddings = embeddings
        self.config.storage.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_cache_dir = self.config.storage.chunk_cache_dir / '_embeddings'
        self._embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    def load_or_create(self, sources: list[SourceText], chunks: list[Document]) -> tuple[str, Path, Chroma]:
        dataset_id = build_dataset_id(sources)
        index_id = self._build_index_id(dataset_id, sources, chunks)
        dataset_dir = self.config.storage.vector_store_dir / f'{dataset_id}_{index_id}'
        manifest_path = dataset_dir / 'manifest.json'
        collection_name = f'edge_rag_{dataset_id}_{index_id}'
        manifest = self._build_manifest(dataset_id, sources, chunks, collection_name)

        with _dataset_lock(dataset_dir):
            current = self._read_manifest(manifest_path)
            if current == manifest:
                store = self._open_store(collection_name, dataset_dir)
                return dataset_id, dataset_dir, store

            self._rebuild_store(dataset_dir, manifest_path, manifest, collection_name, sources, chunks)
            store = self._open_store(collection_name, dataset_dir)
            return dataset_id, dataset_dir, store

    def _build_index_id(self, dataset_id: str, sources: list[SourceText], chunks: list[Document]) -> str:
        payload = {
            'index_format_version': _INDEX_FORMAT_VERSION,
            'dataset_id': dataset_id,
            'embedding_model': self.config.models.embedding_model,
            'chunk_size': self.config.retrieval.chunk_size,
            'chunk_overlap': self.config.retrieval.chunk_overlap,
            'chunk_count': len(chunks),
            'sources': self._source_manifest_entries(sources),
            'chunk_ids': [str(document.metadata.get('chunk_id', '')) for document in chunks],
        }
        encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()[:12]

    def _build_manifest(
        self,
        dataset_id: str,
        sources: list[SourceText],
        chunks: list[Document],
        collection_name: str,
    ) -> dict[str, Any]:
        return {
            'index_format_version': _INDEX_FORMAT_VERSION,
            'dataset_id': dataset_id,
            'collection_name': collection_name,
            'embedding_model': self.config.models.embedding_model,
            'chunk_size': self.config.retrieval.chunk_size,
            'chunk_overlap': self.config.retrieval.chunk_overlap,
            'chunk_count': len(chunks),
            'sources': self._source_manifest_entries(sources),
            'chunk_ids': [str(document.metadata.get('chunk_id', '')) for document in chunks],
        }

    def _read_manifest(self, manifest_path: Path) -> dict[str, Any] | None:
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return None

    def _rebuild_store(
        self,
        dataset_dir: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        collection_name: str,
        sources: list[SourceText],
        chunks: list[Document],
    ) -> None:
        if dataset_dir.exists():
            self._safe_rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = self._open_store(collection_name, dataset_dir)
            embeddings = self._embeddings_for_chunks(sources, chunks)
            self._populate_store(store, chunks, embeddings)
            atomic_write_json(manifest_path, manifest)
        except Exception:
            with suppress(Exception):
                self._safe_rmtree(dataset_dir)
            raise

    def _open_store(self, collection_name: str, dataset_dir: Path) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(dataset_dir),
            embedding_function=self.embeddings,
        )

    def _safe_rmtree(self, target: Path) -> None:
        base = self.config.storage.vector_store_dir.resolve()
        resolved = target.resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f'Refusing to delete path outside managed vector store directory: {resolved}')
        shutil.rmtree(resolved)

    def _source_manifest_entries(self, sources: list[SourceText]) -> list[dict[str, str]]:
        return [
            {
                'name': source.name,
                'path': str(source.path),
                'sha256': source.sha256,
            }
            for source in sorted(sources, key=source_identity_sort_key)
        ]

    def _populate_store(
        self,
        store: Chroma,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        ids = [str(document.metadata['chunk_id']) for document in chunks]
        documents = [document.page_content for document in chunks]
        metadatas = [dict(document.metadata) for document in chunks]
        collection = getattr(store, '_collection', None)
        client = getattr(store, '_client', None)
        if collection is None:
            raise RuntimeError('Chroma collection is not available for vector store population.')

        if client is not None and (hasattr(client, 'get_max_batch_size') or hasattr(client, 'max_batch_size')):
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=client,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
                embeddings=embeddings,
            ):
                collection.upsert(
                    ids=batch[0],
                    embeddings=batch[1],
                    metadatas=batch[2],
                    documents=batch[3],
                )
            return

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def _embeddings_for_chunks(
        self,
        sources: list[SourceText],
        chunks: list[Document],
    ) -> list[list[float]]:
        resolved_embeddings: list[list[float] | None] = [None] * len(chunks)
        chunks_by_source: dict[tuple[str, str], list[tuple[int, Document]]] = {}
        for index, document in enumerate(chunks):
            key = self._source_key_from_document(document)
            chunks_by_source.setdefault(key, []).append((index, document))

        processed_keys: set[tuple[str, str]] = set()
        missing_sources: list[tuple[SourceText, list[tuple[int, Document]]]] = []
        for source in sorted(sources, key=source_identity_sort_key):
            key = (source.sha256, str(source.path))
            if key in processed_keys:
                continue
            processed_keys.add(key)
            source_items = chunks_by_source.get(key, [])
            if not source_items:
                continue
            cached = self._load_cached_source_embeddings(source.sha256, [document for _idx, document in source_items])
            if cached is None:
                missing_sources.append((source, source_items))
                continue
            for (chunk_index, _document), embedding in zip(source_items, cached):
                resolved_embeddings[chunk_index] = embedding

        if missing_sources:
            missing_texts: list[str] = []
            source_ranges: list[tuple[SourceText, list[tuple[int, Document]], int, int]] = []
            for source, source_items in missing_sources:
                start = len(missing_texts)
                missing_texts.extend(document.page_content for _idx, document in source_items)
                source_ranges.append((source, source_items, start, len(source_items)))

            generated_embeddings = self.embeddings.embed_documents(missing_texts)
            for source, source_items, start, count in source_ranges:
                source_embeddings = generated_embeddings[start : start + count]
                for (chunk_index, _document), embedding in zip(source_items, source_embeddings):
                    resolved_embeddings[chunk_index] = embedding
                self._save_cached_source_embeddings(
                    source.sha256,
                    [document for _idx, document in source_items],
                    source_embeddings,
                )

        if any(embedding is None for embedding in resolved_embeddings):
            raise RuntimeError('Failed to resolve embeddings for one or more chunks.')
        return [embedding for embedding in resolved_embeddings if embedding is not None]

    def _load_cached_source_embeddings(
        self,
        source_sha256: str,
        chunks: list[Document],
    ) -> list[list[float]] | None:
        cache_path = self._embedding_cache_path(source_sha256)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get('version') != _EMBEDDING_CACHE_VERSION:
            return None
        if payload.get('source_sha256') != source_sha256:
            return None
        if payload.get('embedding_model') != self.config.models.embedding_model:
            return None
        if payload.get('chunk_size') != int(self.config.retrieval.chunk_size):
            return None
        if payload.get('chunk_overlap') != int(self.config.retrieval.chunk_overlap):
            return None
        cached_chunks = payload.get('chunks')
        if not isinstance(cached_chunks, list):
            return None

        cached_by_id: dict[str, tuple[str, list[float]]] = {}
        for item in cached_chunks:
            if not isinstance(item, dict):
                return None
            chunk_id = item.get('chunk_id')
            content_sha256 = item.get('content_sha256')
            vector = item.get('embedding')
            if not isinstance(chunk_id, str) or not isinstance(content_sha256, str) or not isinstance(vector, list):
                return None
            cached_by_id[chunk_id] = (content_sha256, [float(value) for value in vector])

        resolved: list[list[float]] = []
        for document in chunks:
            chunk_id = str(document.metadata.get('chunk_id', ''))
            cached = cached_by_id.get(chunk_id)
            if cached is None:
                return None
            expected_content_sha256 = self._content_sha256(document.page_content)
            cached_content_sha256, embedding = cached
            if cached_content_sha256 != expected_content_sha256:
                return None
            resolved.append(embedding)
        return resolved

    def _save_cached_source_embeddings(
        self,
        source_sha256: str,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        payload_chunks: list[dict[str, Any]] = []
        for document, embedding in zip(chunks, embeddings):
            payload_chunks.append(
                {
                    'chunk_id': str(document.metadata.get('chunk_id', '')),
                    'chunk_index': int(document.metadata.get('chunk_index', 0)),
                    'content_sha256': self._content_sha256(document.page_content),
                    'embedding': embedding,
                }
            )
        atomic_write_json(
            self._embedding_cache_path(source_sha256),
            {
                'version': _EMBEDDING_CACHE_VERSION,
                'source_sha256': source_sha256,
                'embedding_model': self.config.models.embedding_model,
                'chunk_size': int(self.config.retrieval.chunk_size),
                'chunk_overlap': int(self.config.retrieval.chunk_overlap),
                'chunk_count': len(payload_chunks),
                'chunks': payload_chunks,
            },
        )

    def _embedding_cache_path(self, source_sha256: str) -> Path:
        model_key = hashlib.sha256(self.config.models.embedding_model.encode('utf-8')).hexdigest()[:12]
        return self._embedding_cache_dir / (
            f'{source_sha256}_cs{int(self.config.retrieval.chunk_size)}'
            f'_co{int(self.config.retrieval.chunk_overlap)}_em{model_key}_v{_EMBEDDING_CACHE_VERSION}.json'
        )

    def _source_key_from_document(self, document: Document) -> tuple[str, str]:
        return (
            str(document.metadata.get('source_sha256', '')),
            str(document.metadata.get('source_path', '')),
        )

    def _content_sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
