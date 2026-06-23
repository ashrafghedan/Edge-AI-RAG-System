from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .types import SourceText


def discover_text_files(input_path: str | Path) -> list[Path]:
    path = Path(str(input_path).strip().strip('"').strip("'"))
    if not path.exists():
        raise FileNotFoundError(f'Path does not exist: {path}')
    if path.is_file():
        if path.suffix.lower() != '.txt':
            raise ValueError('Only .txt files are supported for ingestion.')
        return [path.resolve()]
    files = sorted(candidate.resolve() for candidate in path.glob('*.txt') if candidate.is_file())
    if not files:
        raise ValueError('The selected folder does not contain any .txt files.')
    return files


def load_sources(paths: list[Path]) -> list[SourceText]:
    sources: list[SourceText] = []
    for path in paths:
        content = path.read_text(encoding='utf-8', errors='ignore')
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        stat = path.stat()
        sources.append(
            SourceText(
                name=path.name,
                path=path.resolve(),
                content=content,
                sha256=digest,
                size_bytes=stat.st_size,
                modified_at=str(stat.st_mtime),
            )
        )
    return sources


def source_identity_sort_key(source: SourceText) -> tuple[str, str, str]:
    return (source.name.lower(), source.sha256, str(source.path))


def build_dataset_id(sources: list[SourceText]) -> str:
    payload = json.dumps(
        [{'name': item.name, 'sha256': item.sha256} for item in sorted(sources, key=source_identity_sort_key)],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
