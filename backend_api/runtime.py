from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import RLock
import shutil

from edge_rag.app import EdgeRagApp
from edge_rag.config import AppConfig, StorageConfig, default_config
from edge_rag.types import GeneratedQuestion

from .config import ApiSettings, get_settings


class SessionRuntimeManager:
    def __init__(self, settings: ApiSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._base_config = default_config(self.settings.project_root)
        self._apps: dict[str, EdgeRagApp] = {}
        self._lock = RLock()

    def _config_for_session(self, session_id: str) -> AppConfig:
        session_root = self.settings.runtime_dir / session_id
        storage = StorageConfig(
            base_dir=session_root,
            vector_store_dir=self._base_config.storage.vector_store_dir,
            chunk_cache_dir=self._base_config.storage.chunk_cache_dir,
            state_dir=session_root / 'state',
            logs_dir=session_root / 'logs',
            interaction_log_path=session_root / 'logs' / 'interaction_log.json',
            question_history_path=session_root / 'state' / 'question_history.json',
            active_selection_path=session_root / 'state' / 'active_selection.json',
        )
        return replace(self._base_config, storage=storage)

    def get_app(self, session_id: str) -> EdgeRagApp:
        with self._lock:
            app = self._apps.get(session_id)
            if app is None:
                app = EdgeRagApp(config=self._config_for_session(session_id))
                self._apps[session_id] = app
            return app

    def activate_paths(self, session_id: str, paths: list[Path]):
        return self.get_app(session_id).activate_selection(paths, persist=False)

    def activate_sources(self, session_id: str, sources):
        return self.get_app(session_id).activate_sources(sources, persist=False)

    def restore_question(self, session_id: str, generated: GeneratedQuestion) -> None:
        app = self.get_app(session_id)
        app.state.generated_questions[generated.question_id] = generated

    def destroy_session(self, session_id: str) -> None:
        with self._lock:
            self._apps.pop(session_id, None)
        shutil.rmtree(self.settings.runtime_dir / session_id, ignore_errors=True)
        shutil.rmtree(self.settings.uploads_dir / session_id, ignore_errors=True)

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._apps.pop(session_id, None)
        shutil.rmtree(self.settings.runtime_dir / session_id, ignore_errors=True)


_runtime_manager = SessionRuntimeManager()


def get_runtime_manager() -> SessionRuntimeManager:
    return _runtime_manager
