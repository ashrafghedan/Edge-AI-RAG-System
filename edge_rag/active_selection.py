from __future__ import annotations

import json
from pathlib import Path

from .utils import atomic_write_json


class ActiveSelectionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load_paths(self) -> list[Path] | None:
        if not self.path.exists():
            return None

        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            self.clear()
            return None

        raw_paths = payload.get('source_paths')
        if not isinstance(raw_paths, list) or not raw_paths:
            self.clear()
            return None

        resolved: list[Path] = []
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                self.clear()
                return None
            resolved.append(Path(raw_path).resolve())
        return resolved

    def save_paths(self, paths: list[Path]) -> None:
        atomic_write_json(
            self.path,
            {'source_paths': [str(Path(path).resolve()) for path in paths]},
        )

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return

