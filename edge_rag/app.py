from __future__ import annotations

import uuid
from pathlib import Path
from threading import RLock

from .active_selection import ActiveSelectionStore
from .answering import answer_question
from .chatting import ChatResponse, generate_chat_response
from .chunking import build_chunked_documents
from .config import AppConfig, default_config
from .grading import grade_answer
from .json_logger import JsonInteractionLogger
from .llm_client import LlamaCppClients, normalize_inference_error
from .loaders import build_dataset_id, discover_text_files, load_sources
from .policy import get_default_security_policy
from .question_generation import QuestionGenerator
from .question_history import QuestionHistoryStore
from .types import ActiveCorpus, GeneratedQuestion, GroundedAnswerResult, GradingResult, SessionState
from .utils import safe_label
from .vectorstore import ChromaIndexManager
from .types import SourceText


class EdgeRagApp:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or default_config()
        self._prepare_directories()
        self.clients = LlamaCppClients(self.config.models)
        self.history_store = QuestionHistoryStore(self.config.storage.question_history_path)
        self.selection_store = ActiveSelectionStore(self.config.storage.active_selection_path)
        self.logger = JsonInteractionLogger(self.config.storage.interaction_log_path)
        self.security_policy = get_default_security_policy()
        self.question_generator = QuestionGenerator(self.config, self.history_store, self.security_policy)
        self.state = SessionState(session_id=uuid.uuid4().hex)
        self._vector_store = None
        self._index_manager = ChromaIndexManager(self.config, self.clients.embeddings)
        self._activation_lock = RLock()

    def system_status(self) -> tuple[bool, str]:
        return self.clients.check_server()

    def discover_sources(self, input_path: str | Path) -> list[Path]:
        return discover_text_files(input_path)

    def activate_selection(self, selected_paths: list[Path], *, persist: bool = True) -> ActiveCorpus:
        with self._activation_lock:
            resolved_paths = [Path(path).resolve() for path in selected_paths]
            sources = load_sources(resolved_paths)
            return self._activate_loaded_sources(sources, persist=persist, persist_paths=resolved_paths)

    def activate_sources(self, sources: list[SourceText], *, persist: bool = True) -> ActiveCorpus:
        with self._activation_lock:
            resolved_paths = [Path(source.path).resolve() for source in sources]
            return self._activate_loaded_sources(sources, persist=persist, persist_paths=resolved_paths)

    def restore_last_selection(self) -> ActiveCorpus | None:
        selected_paths = self.selection_store.load_paths()
        if not selected_paths:
            return None

        if not all(path.exists() and path.is_file() and path.suffix.lower() == '.txt' for path in selected_paths):
            self.selection_store.clear()
            return None

        try:
            return self.activate_selection(selected_paths, persist=False)
        except Exception:
            self.selection_store.clear()
            return None

    def ask_question(self, question: str) -> GroundedAnswerResult:
        active = self.require_active_corpus()
        query_decision = self.security_policy.evaluate_text(question, purpose='user_query')
        if query_decision.blocked:
            self.logger.log_security_event(
                self.state.session_id,
                active.dataset_id,
                stage='user_question',
                text_preview=self.security_policy.audit_preview(question),
                decision=query_decision,
                source_file_names=active.source_names,
            )
            result = GroundedAnswerResult(
                answer=self.security_policy.refusal_message(query_decision),
                found=False,
                source_names=[],
                evidence_ids=[],
                retrieved_chunks=[],
            )
            self.logger.log_user_question(self.state.session_id, active.dataset_id, question, result)
            return result

        try:
            result = answer_question(
                self.clients.answer_llm,
                self._vector_store,
                self.config,
                question,
                source_chunks=active.chunks,
            )
        except Exception as exc:
            normalized = normalize_inference_error(exc, base_url=self.clients.config.ollama_base_url,)
            if normalized is not None:
                raise normalized from exc
            raise
        answer_decision = self.security_policy.evaluate_text(result.answer, purpose='model_output')
        if result.answer != self.config.not_available_response and answer_decision.blocked:
            self.logger.log_security_event(
                self.state.session_id,
                active.dataset_id,
                stage='model_answer',
                text_preview=self.security_policy.audit_preview(result.answer),
                decision=answer_decision,
                source_file_names=result.source_names or active.source_names,
            )
            result = GroundedAnswerResult(
                answer=self.security_policy.refusal_message(answer_decision),
                found=False,
                source_names=result.source_names,
                evidence_ids=[],
                retrieved_chunks=result.retrieved_chunks,
            )
        self.logger.log_user_question(self.state.session_id, active.dataset_id, question, result)
        return result

    def chat(
        self,
        message: str,
        *,
        history: list[dict[str, object]] | None = None,
        attachments: list[dict[str, object]] | None = None,
    ) -> str:
        return self.chat_result(message, history=history, attachments=attachments).answer

    def chat_result(
        self,
        message: str,
        *,
        history: list[dict[str, object]] | None = None,
        attachments: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        try:
            return generate_chat_response(
                self.clients,
                message,
                history=history,
                attachments=attachments,
                security_policy=self.security_policy,
            )
        except Exception as exc:
            normalized = normalize_inference_error(exc, base_url=self.clients.config.ollama_base_url,)
            if normalized is not None:
                raise normalized from exc
            raise

    def generate_question(self) -> GeneratedQuestion:
        active = self.require_active_corpus()
        try:
            generated = self.question_generator.generate(
                self.clients.question_llm,
                self.clients.answer_llm,
                self._vector_store,
                active.dataset_id,
                self.state.session_id,
                active.chunks,
            )
        except Exception as exc:
            normalized = normalize_inference_error(exc, base_url=self.clients.config.ollama_base_url,)
            if normalized is not None:
                raise normalized from exc
            raise
        self.state.generated_questions[generated.question_id] = generated
        self.logger.log_generated_question(self.state.session_id, active.dataset_id, generated)
        return generated

    def grade_generated_question(self, question_id: str, user_answer: str) -> GradingResult:
        active = self.require_active_corpus()
        generated = self.state.generated_questions.get(question_id)
        if generated is None:
            raise KeyError(f'Unknown generated question id: {question_id}')
        answer_decision = self.security_policy.evaluate_text(user_answer, purpose='user_answer')
        if answer_decision.blocked:
            self.logger.log_security_event(
                self.state.session_id,
                active.dataset_id,
                stage='user_answer',
                text_preview=self.security_policy.audit_preview(user_answer),
                decision=answer_decision,
                source_file_names=generated.source_names,
            )
            grading = GradingResult(
                score=0,
                feedback=self.security_policy.refusal_message(answer_decision),
                model_answer='Hidden by local security policy.',
            )
            self.logger.log_grading(self.state.session_id, active.dataset_id, generated, user_answer, grading)
            return grading
        try:
            grading = grade_answer(
                self.clients.grading_llm,
                self.clients.embeddings,
                self._vector_store,
                self.config,
                generated,
                user_answer,
                source_chunks=active.chunks,
            )
        except Exception as exc:
            normalized = normalize_inference_error(exc, base_url=self.clients.config.ollama_base_url,)
            if normalized is not None:
                raise normalized from exc
            raise
        grading_decision = self.security_policy.evaluate_text(
            f'{grading.feedback}\n{grading.model_answer}',
            purpose='model_output',
        )
        if grading_decision.blocked:
            self.logger.log_security_event(
                self.state.session_id,
                active.dataset_id,
                stage='grading_output',
                text_preview=self.security_policy.audit_preview(f'{grading.feedback} {grading.model_answer}'),
                decision=grading_decision,
                source_file_names=generated.source_names,
            )
            grading = GradingResult(
                score=grading.score,
                feedback=self.security_policy.refusal_message(grading_decision),
                model_answer='Hidden by local security policy.',
            )
        self.logger.log_grading(self.state.session_id, active.dataset_id, generated, user_answer, grading)
        return grading

    def require_active_corpus(self) -> ActiveCorpus:
        if self.state.active_corpus is None or self._vector_store is None:
            raise RuntimeError('No active corpus is loaded. Load a .txt file or folder first.')
        return self.state.active_corpus

    def _prepare_directories(self) -> None:
        self.config.storage.base_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.logs_dir.mkdir(parents=True, exist_ok=True)

    def _activate_loaded_sources(
        self,
        sources: list[SourceText],
        *,
        persist: bool,
        persist_paths: list[Path],
    ) -> ActiveCorpus:
        if self._active_corpus_matches(sources):
            active = self.require_active_corpus()
            if persist:
                self.selection_store.save_paths(persist_paths)
            return active

        chunks = build_chunked_documents(
            sources,
            self.config.retrieval,
            cache_dir=self.config.storage.chunk_cache_dir,
        )
        dataset_id, vector_directory, vector_store = self._index_manager.load_or_create(sources, chunks)
        active = ActiveCorpus(
            dataset_id=dataset_id,
            dataset_label=safe_label([source.name for source in sources]),
            source_names=[source.name for source in sources],
            source_paths=[str(source.path) for source in sources],
            vector_directory=vector_directory,
            chunk_count=len(chunks),
            chunks=chunks,
        )
        self.state.active_corpus = active
        self.state.generated_questions.clear()
        self._vector_store = vector_store
        if persist:
            self.selection_store.save_paths(persist_paths)
        return active

    def _active_corpus_matches(self, sources: list[SourceText]) -> bool:
        if self.state.active_corpus is None or self._vector_store is None:
            return False
        active = self.state.active_corpus
        target_dataset_id = build_dataset_id(sources)
        if active.dataset_id != target_dataset_id:
            return False
        active_paths = {str(Path(path).resolve()) for path in active.source_paths}
        target_paths = {str(Path(source.path).resolve()) for source in sources}
        return active_paths == target_paths
