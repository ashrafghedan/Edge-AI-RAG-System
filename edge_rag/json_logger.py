from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .policy import SecurityDecision
from .types import GeneratedQuestion, GradingResult, GroundedAnswerResult
from .utils import atomic_write_json, utc_now_iso


class JsonInteractionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            atomic_write_json(self.path, {'entries': []})

    def log_user_question(
        self,
        session_id: str,
        dataset_id: str,
        question: str,
        result: GroundedAnswerResult,
    ) -> None:
        self._append(
            {
                'event_type': 'user_question',
                'timestamp': utc_now_iso(),
                'session_id': session_id,
                'dataset_id': dataset_id,
                'source_file_names': result.source_names,
                'user_question': question,
                'generated_question': None,
                'model_answer': result.answer,
                'user_answer': None,
                'score': None,
                'feedback': None,
                'evidence_ids': result.evidence_ids,
            }
        )

    def log_generated_question(
        self,
        session_id: str,
        dataset_id: str,
        generated: GeneratedQuestion,
    ) -> None:
        self._append(
            {
                'event_type': 'generated_question',
                'timestamp': generated.created_at,
                'session_id': session_id,
                'dataset_id': dataset_id,
                'source_file_names': generated.source_names,
                'user_question': None,
                'generated_question': generated.question,
                'model_answer': generated.model_answer,
                'user_answer': None,
                'score': None,
                'feedback': None,
                'question_id': generated.question_id,
                'source_chunk_ids': generated.source_chunk_ids,
            }
        )

    def log_grading(
        self,
        session_id: str,
        dataset_id: str,
        generated: GeneratedQuestion,
        user_answer: str,
        grading: GradingResult,
    ) -> None:
        self._append(
            {
                'event_type': 'graded_answer',
                'timestamp': utc_now_iso(),
                'session_id': session_id,
                'dataset_id': dataset_id,
                'source_file_names': generated.source_names,
                'user_question': None,
                'generated_question': generated.question,
                'model_answer': grading.model_answer,
                'user_answer': user_answer,
                'score': grading.score,
                'feedback': grading.feedback,
                'question_id': generated.question_id,
                'source_chunk_ids': generated.source_chunk_ids,
            }
        )

    def log_security_event(
        self,
        session_id: str,
        dataset_id: str | None,
        *,
        stage: str,
        text_preview: str,
        decision: SecurityDecision,
        source_file_names: list[str] | None = None,
    ) -> None:
        self._append(
            {
                'event_type': 'security_event',
                'timestamp': utc_now_iso(),
                'session_id': session_id,
                'dataset_id': dataset_id,
                'source_file_names': source_file_names or [],
                'security_stage': stage,
                'security_blocked': decision.blocked,
                'security_categories': decision.categories,
                'security_reasons': [finding.reason for finding in decision.findings],
                'text_preview': text_preview,
            }
        )

    def _append(self, entry: dict[str, Any]) -> None:
        payload = json.loads(self.path.read_text(encoding='utf-8'))
        payload.setdefault('entries', []).append(entry)
        atomic_write_json(self.path, payload)
