from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .utils import atomic_write_json, normalize_question_text, utc_now_iso


class QuestionHistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def recent_questions(self, dataset_id: str, limit: int = 10) -> list[str]:
        records = self._data.get('datasets', {}).get(dataset_id, [])
        recent = [record['question'] for record in records if record.get('normalized')]
        return recent[-limit:]

    def is_duplicate(self, dataset_id: str, question: str) -> bool:
        normalized = normalize_question_text(question)
        if not normalized:
            return False
        for record in self._data.get('datasets', {}).get(dataset_id, []):
            existing = record.get('normalized', '')
            if existing == normalized:
                return True
            if self._token_overlap(existing, normalized) >= 0.82:
                return True
        return False

    def record(self, dataset_id: str, question: str, session_id: str, source_names: list[str]) -> None:
        normalized = normalize_question_text(question)
        if not normalized:
            return
        datasets = self._data.setdefault('datasets', {})
        records = datasets.setdefault(dataset_id, [])
        records.append(
            {
                'question': question,
                'normalized': normalized,
                'session_id': session_id,
                'source_names': source_names,
                'created_at': utc_now_iso(),
            }
        )
        atomic_write_json(self.path, self._data)

    def prune_questions(self, should_remove: Callable[[str], bool], dataset_id: str | None = None) -> int:
        datasets = self._data.get('datasets', {})
        removed = 0
        target_ids = [dataset_id] if dataset_id is not None else list(datasets.keys())
        for current_dataset_id in target_ids:
            records = datasets.get(current_dataset_id, [])
            kept: list[dict] = []
            for record in records:
                question = str(record.get('question', ''))
                if should_remove(question):
                    removed += 1
                    continue
                kept.append(record)
            datasets[current_dataset_id] = kept
        if removed:
            atomic_write_json(self.path, self._data)
        return removed

    def _load(self) -> dict:
        if not self.path.exists():
            return {'datasets': {}}
        return json.loads(self.path.read_text(encoding='utf-8'))

    @staticmethod
    def _token_overlap(left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
