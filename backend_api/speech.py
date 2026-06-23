from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from edge_rag.utils import normalize_whitespace

from .config import ApiSettings, get_settings


class SpeechToTextError(RuntimeError):
    pass


class SpeechToTextUnavailableError(SpeechToTextError):
    pass


@dataclass(slots=True)
class SpeechToTextResult:
    text: str
    language: str | None = None


class SpeechToTextService:
    def __init__(self, settings: ApiSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._engine = None
        self._lock = RLock()

    def transcribe_file(self, audio_path: Path, *, language: str | None = None) -> SpeechToTextResult:
        engine = self._get_engine()
        normalized_language = _normalize_language(language or self.settings.stt_default_language)
        return engine.transcribe(audio_path, language=normalized_language)

    def _get_engine(self):
        with self._lock:
            if self._engine is None:
                self._engine = _build_engine(self.settings)
            return self._engine


class _WhisperTorchEngine:
    def __init__(self, settings: ApiSettings) -> None:
        try:
            import torch
            import whisper
        except ImportError as exc:  # pragma: no cover - depends on runtime installation
            raise SpeechToTextUnavailableError(
                'Speech-to-text needs the local `openai-whisper` package and a working PyTorch installation.'
            ) from exc

        self._torch = torch
        self._whisper = whisper
        self._device = _resolve_torch_device(settings.stt_device, torch_module=torch)
        model_name = _resolve_model_reference(settings)
        try:
            self._model = whisper.load_model(model_name, device=self._device)
        except Exception as exc:  # pragma: no cover - depends on runtime installation
            raise SpeechToTextUnavailableError(
                f'Unable to load Whisper model `{settings.stt_model}` on device `{self._device}`. '
                'For fully offline use, set `EDGE_RAG_STT_MODEL` to a local model directory or make sure the named model is already cached on disk.'
            ) from exc

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> SpeechToTextResult:
        try:
            payload = self._model.transcribe(
                str(audio_path),
                language=language,
                fp16=self._device == 'cuda',
                verbose=False,
                temperature=0.0,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise SpeechToTextError(f'Speech transcription failed: {exc}') from exc

        text = normalize_whitespace(str(payload.get('text') or ''))
        if not text:
            raise SpeechToTextError('No speech could be transcribed from the provided audio.')
        detected_language = _normalize_language(str(payload.get('language') or '') or language)
        return SpeechToTextResult(text=text, language=detected_language)


class _FasterWhisperEngine:
    def __init__(self, settings: ApiSettings) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on runtime installation
            raise SpeechToTextUnavailableError(
                'Speech-to-text needs either local `openai-whisper` or local `faster-whisper` installed.'
            ) from exc

        device = _resolve_basic_device(settings.stt_device)
        compute_type = settings.stt_compute_type or ('float16' if device == 'cuda' else 'int8')
        model_name = _resolve_model_reference(settings)
        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:  # pragma: no cover - depends on runtime installation
            raise SpeechToTextUnavailableError(
                f'Unable to load faster-whisper model `{settings.stt_model}` on device `{device}`. '
                'For fully offline use, set `EDGE_RAG_STT_MODEL` to a local model directory or make sure the named model is already cached on disk.'
            ) from exc

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> SpeechToTextResult:
        try:
            segments, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = normalize_whitespace(' '.join(segment.text.strip() for segment in segments if segment.text.strip()))
        except Exception as exc:
            raise SpeechToTextError(f'Speech transcription failed: {exc}') from exc

        if not text:
            raise SpeechToTextError('No speech could be transcribed from the provided audio.')
        return SpeechToTextResult(text=text, language=_normalize_language(getattr(info, 'language', None) or language))


def _build_engine(settings: ApiSettings):
    backend = settings.stt_backend
    candidates = _backend_candidates(backend)
    errors: list[str] = []
    for candidate in candidates:
        try:
            if candidate == 'whisper':
                return _WhisperTorchEngine(settings)
            if candidate == 'faster_whisper':
                return _FasterWhisperEngine(settings)
        except SpeechToTextUnavailableError as exc:
            errors.append(str(exc))
            continue
    suffix = f" Tried: {'; '.join(errors)}" if errors else ''
    raise SpeechToTextUnavailableError(
        'No speech-to-text backend is available. Offline speech-to-text requires local `openai-whisper` with PyTorch support, or local `faster-whisper`.'
        + suffix
    )


def _backend_candidates(configured_backend: str) -> list[str]:
    if configured_backend == 'whisper':
        return ['whisper']
    if configured_backend == 'faster_whisper':
        return ['faster_whisper']
    if platform.machine().lower() in {'aarch64', 'arm64'}:
        return ['faster_whisper', 'whisper']
    return ['whisper', 'faster_whisper']


def _resolve_torch_device(configured_device: str, *, torch_module) -> str:
    if configured_device in {'cpu', 'cuda'}:
        return configured_device
    return 'cuda' if getattr(torch_module.cuda, 'is_available', lambda: False)() else 'cpu'


def _resolve_basic_device(configured_device: str) -> str:
    if configured_device in {'cpu', 'cuda'}:
        return configured_device
    return 'cpu'


def _resolve_model_reference(settings: ApiSettings) -> str:
    configured = str(settings.stt_model or '').strip()
    if not configured:
        return configured

    candidate = Path(configured).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())

    relative = (settings.project_root / configured).resolve()
    if relative.exists():
        return str(relative)

    if candidate.exists():
        return str(candidate.resolve())

    return configured


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    lowered = str(language).strip().lower()
    if not lowered:
        return None
    if lowered.startswith('ar'):
        return 'ar'
    if lowered.startswith('en'):
        return 'en'
    return lowered


_speech_service = SpeechToTextService()


def get_transcription_service() -> SpeechToTextService:
    return _speech_service
