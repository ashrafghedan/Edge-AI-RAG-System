"""HTTP client for a locally hosted llama.cpp server.

Talks to the OpenAI-compatible endpoints exposed by ``llama-server``:

* ``GET  /v1/models``               - readiness probe
* ``POST /v1/chat/completions``     - non-streaming and streaming chat
* ``POST /v1/embeddings``           - embeddings for RAG

The class surface mirrors the older Ollama client so the rest of the
application keeps the same call sites (chat, RAG answering, grading,
question generation, embeddings, etc.).
"""

from __future__ import annotations

import json
import re
import socket
import time
from dataclasses import dataclass
from http.client import RemoteDisconnected
from typing import Any, Iterator
import urllib.error
import urllib.request

from langchain_core.embeddings import Embeddings

from .config import ModelConfig
from .response_validation import finalize_response_text, response_needs_continuation


_CHAT_CONTINUATION_PROMPT = (
    'Continue exactly from where you stopped. '
    'Do not repeat earlier text. '
    'Continue toward the natural ending of the requested answer. '
    'Finish any open sentence, paragraph, list, code block, JSON block, or story ending cleanly.'
)
_FINAL_ANSWER_ONLY_PROMPT = (
    'Provide only the final answer to the original user request now. '
    'Do not include your reasoning, analysis, or thinking process. '
    'Answer directly and completely.'
)
_JSON_CONTINUATION_PROMPT = (
    'Continue the same JSON response from the exact stopping point. '
    'Do not restart the object, do not wrap it in markdown, and do not repeat earlier text. '
    'Finish any open strings, arrays, objects, and sentences, then stop.'
)
_RESOURCE_ERROR_SIGNALS = (
    'out of memory',
    'not enough memory',
    'insufficient memory',
    'requires more system memory',
    'cuda error',
    'cuda out of memory',
    'no kv slot available',
    'context length',
)
_VISION_ERROR_SIGNALS = (
    'image input is not supported',
    'image input is unsupported',
    'no image support',
    'model does not support image',
    'no multimodal projector',
)
_EMBEDDING_POOLING_ERROR_SIGNALS = (
    "pooling type 'none' is not oai compatible",
    'pooling type "none" is not oai compatible',
)
_ATTACHMENT_CONTEXT_MARKER = 'Use the attached material below when answering.'
_SHORT_ANSWER_HINTS = (
    'brief',
    'briefly',
    'concise',
    'short answer',
    'short explanation',
    'in one sentence',
    'in 1 sentence',
    'one sentence',
    'two sentences',
    'three sentences',
    'tl;dr',
)
_LONG_ANSWER_HINTS = (
    'comprehensive',
    'deep dive',
    'detailed',
    'full explanation',
    'in depth',
    'long answer',
    'step by step',
    'thorough',
    'very long',
    'walk me through',
)
_LONG_FORM_NOUNS = (
    'article',
    'chapter',
    'essay',
    'fiction',
    'guide',
    'narrative',
    'question',
    'answer',
    'response',
    'story',
    'tutorial',
)
_LIST_OR_COMPARE_HINTS = (
    'compare',
    'difference between',
    'pros and cons',
    'summarize',
    'summary',
    'top ',
)
_CODE_REQUEST_HINTS = (
    'api',
    'bash',
    'bug',
    'class ',
    'code',
    'command',
    'css',
    'debug',
    'example',
    'function',
    'html',
    'javascript',
    'json',
    'program',
    'python',
    'regex',
    'script',
    'snippet',
    'sql',
    'yaml',
)
_SIMPLE_QUESTION_PREFIXES = (
    'can ',
    'could ',
    'does ',
    'how ',
    'is ',
    'what ',
    'when ',
    'where ',
    'which ',
    'who ',
    'why ',
)
_CHAT_CONTINUATION_TAIL_CHARS = 3200
_MIN_CHAT_NUM_PREDICT = 192
_MAX_CHAT_CONTINUATIONS = 8
_MIN_REQUEST_TIMEOUT_SECONDS = 30
_MAX_REQUEST_TIMEOUT_SECONDS = 900


class LlamaCppInferenceError(RuntimeError):
    """Base error for failures while talking to the llama.cpp server."""


class LlamaCppResourceLimitError(LlamaCppInferenceError):
    """Server ran out of GPU/CPU memory or context room."""


class LlamaCppVisionModelError(LlamaCppInferenceError):
    """Selected model cannot process image input."""


# Backwards-compatible aliases so callers that already imported the old
# Ollama-named errors keep working until the migration is fully complete.
OllamaInferenceError = LlamaCppInferenceError
OllamaResourceLimitError = LlamaCppResourceLimitError
OllamaVisionModelError = LlamaCppVisionModelError


@dataclass(slots=True)
class _ModelResponse:
    content: str


@dataclass(slots=True)
class StreamChunk:
    """Single event emitted by :meth:`LlamaCppClients.stream_chat_completion`.

    ``kind`` values:
        * ``'token'``      - assistant content delta (final answer)
        * ``'reasoning'``  - reasoning/thinking delta extracted by llama-server
                             when ``--reasoning-format deepseek`` is active
        * ``'meta'``       - periodic stats snapshot (tokens, tok/s, elapsed)
        * ``'done'``       - terminal event with final stats + assembled answer
        * ``'error'``      - terminal error message
    """

    kind: str
    text: str = ''
    payload: dict[str, Any] | None = None


class _ValidatedLlamaChatModel:
    """Adapter that mimics the langchain-style ``.invoke(messages)`` surface."""

    def __init__(
        self,
        owner: 'LlamaCppClients',
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        response_format: str | None = None,
    ) -> None:
        self._owner = owner
        self._model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._response_format = response_format

    def invoke(self, messages: list[dict[str, Any]]) -> _ModelResponse:
        content = self._owner._complete_text(
            messages,
            model=self._model,
            temperature=self._temperature,
            num_ctx=self._num_ctx,
            num_predict=self._num_predict,
            response_format=self._response_format,
        )
        return _ModelResponse(content=content)


def normalize_inference_error(exc: Exception, *, base_url: str) -> LlamaCppInferenceError | None:
    if isinstance(exc, LlamaCppInferenceError):
        return exc

    if isinstance(exc, urllib.error.HTTPError):
        detail = _http_error_detail(exc)
    elif isinstance(
        exc,
        (urllib.error.URLError, TimeoutError, socket.timeout, RemoteDisconnected, ConnectionError),
    ):
        reason = getattr(exc, 'reason', None)
        suffix = str(reason or exc).strip()
        message = (
            f'Could not reach llama.cpp server at {base_url}. '
            'Make sure llama-server is running (npm run dev starts it automatically).'
        )
        if suffix:
            message = f'{message} {suffix}'
        return LlamaCppInferenceError(message)
    else:
        detail = str(exc).strip()
        if 'llama' not in detail.lower() and 'model' not in detail.lower():
            return None

    if _looks_like_resource_issue(detail):
        message = (
            'llama.cpp ran out of resources while generating a response. '
            'Lower LLAMA_CPP_NUM_CTX / LLAMA_CPP_MAX_TOKENS or restart llama-server.'
        )
        if detail:
            message = f'{message} Server said: {detail}'
        return LlamaCppResourceLimitError(message)

    if _looks_like_embedding_pooling_issue(detail):
        return LlamaCppInferenceError(
            'llama.cpp embeddings are enabled, but the server is using pooling=none. '
            'Restart llama-server with an OpenAI-compatible pooling mode such as `--pooling mean`.'
        )

    if _looks_like_vision_model_issue(detail):
        return LlamaCppVisionModelError(
            'The currently loaded llama.cpp model is text-only and cannot read images. '
            'Restart llama-server with a multimodal model (and --mmproj) if you need image input.'
        )

    if not detail:
        detail = 'llama.cpp returned an empty error response.'
    return LlamaCppInferenceError(f'llama.cpp failed to generate a response. {detail}')


# Legacy alias used by older callers.
normalize_ollama_error = normalize_inference_error


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    body = ''
    try:
        body = exc.read().decode('utf-8', errors='replace').strip()
    except Exception:
        body = ''
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get('error')
            if isinstance(error, dict):
                message = str(error.get('message') or '').strip()
                if message:
                    return message
            elif isinstance(error, str):
                if error.strip():
                    return error.strip()
            message = str(payload.get('message') or '').strip()
            if message:
                return message
        return body
    return f'HTTP {exc.code}: {exc.reason}'


def _looks_like_resource_issue(detail: str) -> bool:
    lowered = detail.lower()
    return any(signal in lowered for signal in _RESOURCE_ERROR_SIGNALS)


def _looks_like_vision_model_issue(detail: str) -> bool:
    lowered = detail.lower()
    return any(signal in lowered for signal in _VISION_ERROR_SIGNALS)


def _looks_like_embedding_pooling_issue(detail: str) -> bool:
    lowered = detail.lower()
    return any(signal in lowered for signal in _EMBEDDING_POOLING_ERROR_SIGNALS)


def _has_image_input(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get('images'):
            return True
    return False


class LlamaCppEmbeddings(Embeddings):
    """Embeddings client backed by llama-server's ``/v1/embeddings`` endpoint."""

    def __init__(self, *, base_url: str, model: str, timeout: float = 60.0, batch_size: int = 24) -> None:
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._timeout = float(timeout)
        self._batch_size = max(1, int(batch_size))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_many(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_many([text])[0]

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        normalized_inputs = [text or ' ' for text in texts]
        payload = {
            'model': self._model,
            'input': normalized_inputs if len(normalized_inputs) > 1 else normalized_inputs[0],
        }
        request = urllib.request.Request(
            f'{self._base_url}/v1/embeddings',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode('utf-8'))
        except Exception as exc:
            normalized = normalize_inference_error(exc, base_url=self._base_url)
            if normalized is not None:
                raise normalized from exc
            raise

        data = body.get('data') or []
        if not data or not isinstance(data, list):
            raise LlamaCppInferenceError(
                'llama.cpp embeddings response did not contain any vectors. '
                'Make sure llama-server was started with --embeddings.'
            )
        vectors_by_index: dict[int, list[float]] = {}
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise LlamaCppInferenceError('Malformed llama.cpp embeddings response.')
            index = item.get('index')
            resolved_index = index if isinstance(index, int) else fallback_index
            vector = item.get('embedding')
            if not isinstance(vector, list):
                raise LlamaCppInferenceError('Malformed llama.cpp embeddings vector.')
            vectors_by_index[resolved_index] = [float(value) for value in vector]
        vectors = [vectors_by_index[index] for index in range(len(normalized_inputs)) if index in vectors_by_index]
        if len(vectors) != len(normalized_inputs):
            raise LlamaCppInferenceError(
                'llama.cpp embeddings response returned an unexpected number of vectors.'
            )
        return vectors


class LlamaCppClients:
    """Aggregate of llama.cpp-backed chat/answer/grading/question/embedding clients."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._chat_llm: _ValidatedLlamaChatModel | None = None
        self._answer_llm: _ValidatedLlamaChatModel | None = None
        self._question_llm: _ValidatedLlamaChatModel | None = None
        self._grading_llm: _ValidatedLlamaChatModel | None = None
        self._embeddings: LlamaCppEmbeddings | None = None

    # -- Status -----------------------------------------------------------

    def check_server(self) -> tuple[bool, str]:
        names, error = self.available_models()
        if error is not None:
            url = f'{self.config.ollama_base_url}/v1/models'
            return False, (
                f'llama.cpp is not reachable at {url}. Start it with `npm run dev` or run '
                f'llama-server manually. ({error})'
            )
        loaded = ', '.join(filter(None, names)) or '(unnamed)'
        if self.config.answer_model not in names:
            return False, (
                'llama.cpp is online, but the configured answer model is not loaded. '
                f'Expected `{self.config.answer_model}`, but the server reported: {loaded}. '
                'Restart llama-server so the configured model is active.'
            )
        if self.config.embedding_model and self.config.embedding_model not in names:
            return False, (
                'llama.cpp is online, but the configured embedding model is not loaded. '
                f'Expected `{self.config.embedding_model}`, but the server reported: {loaded}. '
                'Restart llama-server so the configured model is active.'
            )
        return True, f'llama.cpp server is online. Models: {loaded}.'

    def available_models(self) -> tuple[list[str], Exception | None]:
        url = f'{self.config.ollama_base_url}/v1/models'
        request = urllib.request.Request(url, method='GET')
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return [], exc
        models = payload.get('data') if isinstance(payload, dict) else None
        names = [str(item.get('id') or '') for item in (models or []) if isinstance(item, dict)]
        return names, None

    # -- Lazy adapters used by the higher level edge_rag layer ------------

    @property
    def answer_llm(self) -> _ValidatedLlamaChatModel:
        if self._answer_llm is None:
            self._answer_llm = _ValidatedLlamaChatModel(
                self,
                model=self.config.answer_model,
                temperature=self.config.answer_temperature,
                num_ctx=self.config.num_ctx,
                num_predict=self.config.max_answer_tokens,
                response_format='json',
            )
        return self._answer_llm

    @property
    def chat_llm(self) -> _ValidatedLlamaChatModel:
        if self._chat_llm is None:
            self._chat_llm = _ValidatedLlamaChatModel(
                self,
                model=self.config.answer_model,
                temperature=max(self.config.answer_temperature, 0.2),
                num_ctx=self.config.num_ctx,
                num_predict=self.config.max_answer_tokens,
            )
        return self._chat_llm

    @property
    def question_llm(self) -> _ValidatedLlamaChatModel:
        if self._question_llm is None:
            self._question_llm = _ValidatedLlamaChatModel(
                self,
                model=self.config.answer_model,
                temperature=self.config.question_temperature,
                num_ctx=min(self.config.num_ctx, 2048),
                num_predict=min(self.config.max_answer_tokens, 160),
                response_format='json',
            )
        return self._question_llm

    @property
    def grading_llm(self) -> _ValidatedLlamaChatModel:
        if self._grading_llm is None:
            self._grading_llm = _ValidatedLlamaChatModel(
                self,
                model=self.config.answer_model,
                temperature=self.config.grading_temperature,
                num_ctx=min(self.config.num_ctx, 2048),
                num_predict=min(self.config.max_grading_tokens, 180),
                response_format='json',
            )
        return self._grading_llm

    @property
    def embeddings(self) -> LlamaCppEmbeddings:
        if self._embeddings is None:
            self._embeddings = LlamaCppEmbeddings(
                base_url=self.config.embedding_base_url or self.config.ollama_base_url,
                model=self.config.embedding_model,
                timeout=self.config.request_timeout_seconds,
                batch_size=self.config.embedding_batch_size,
            )
        return self._embeddings

    # -- Chat (free-form) -------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        use_vision_model: bool = False,
    ) -> str:
        return self._complete_text(
            messages,
            model=self.config.answer_model,
            temperature=max(self.config.answer_temperature, 0.2),
            num_ctx=self.config.num_ctx,
            num_predict=self._chat_num_predict(messages, use_vision_model=use_vision_model),
            response_format=None,
            max_continuations=self._chat_continuation_limit(messages, use_vision_model=use_vision_model),
        )

    def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        use_vision_model: bool = False,
    ) -> Iterator[StreamChunk]:
        """Yield streaming chunks for the given chat messages.

        Yields ``StreamChunk(kind='token', text=...)`` for every received token
        segment, ``StreamChunk(kind='meta', payload=...)`` for periodic stats
        snapshots and finally ``StreamChunk(kind='done', payload=stats)``.
        """

        # llama-server (with --mmproj) supports streaming image+text input via
        # OpenAI-compatible chat completions, so we no longer block here.
        base_conversation = self._normalize_messages(messages)
        conversation = list(base_conversation)
        max_continuations = self._chat_continuation_limit(messages, use_vision_model=use_vision_model)
        num_predict = self._chat_num_predict(messages, use_vision_model=use_vision_model)
        temperature = max(self.config.answer_temperature, 0.2)

        answer = ''
        reasoning = ''
        continuations_used = 0
        final_answer_retry_used = False
        prompt_eval_count = 0
        eval_count = 0
        reasoning_token_count = 0
        started_at = time.monotonic()
        last_meta_emitted = started_at

        while True:
            segment_text = ''
            finish_reason: str | None = None

            try:
                for sse_event in self._iter_chat_completion_stream(
                    conversation,
                    model=self.config.answer_model,
                    temperature=temperature,
                    num_ctx=self.config.num_ctx,
                    num_predict=num_predict,
                    response_format=None,
                ):
                    event_type = sse_event.get('type')
                    if event_type == 'reasoning':
                        text = str(sse_event.get('text') or '')
                        if not text:
                            continue
                        reasoning += text
                        reasoning_token_count += 1
                        yield StreamChunk(kind='reasoning', text=text)
                    elif event_type == 'token':
                        text = str(sse_event.get('text') or '')
                        if not text:
                            continue
                        segment_text += text
                        answer += text
                        eval_count += 1
                        yield StreamChunk(kind='token', text=text)
                        now = time.monotonic()
                        if now - last_meta_emitted >= 0.5:
                            last_meta_emitted = now
                            elapsed = max(1e-3, now - started_at)
                            yield StreamChunk(
                                kind='meta',
                                payload={
                                    'tokens': eval_count,
                                    'reasoning_tokens': reasoning_token_count,
                                    'elapsed_seconds': round(elapsed, 3),
                                    'tokens_per_second': round(eval_count / elapsed, 2),
                                },
                            )
                    elif event_type == 'usage':
                        usage = sse_event.get('usage') or {}
                        prompt_eval_count = int(usage.get('prompt_tokens') or prompt_eval_count)
                        if usage.get('completion_tokens'):
                            eval_count = int(usage['completion_tokens'])
                    elif event_type == 'finish':
                        finish_reason = str(sse_event.get('finish_reason') or '').strip().lower()
            except Exception as exc:
                normalized = normalize_inference_error(
                    exc,
                    base_url=self.config.ollama_base_url,
                )
                if normalized is not None:
                    yield StreamChunk(kind='error', text=str(normalized))
                    return
                raise

            if not segment_text and not finish_reason:
                # Server closed the stream without yielding anything useful.
                break

            if not answer.strip():
                # Some reasoning-enabled runs exhaust themselves inside the
                # thinking phase. Give the model one focused retry that asks
                # for only the final answer before surfacing a fallback.
                if reasoning.strip() and not final_answer_retry_used:
                    final_answer_retry_used = True
                    conversation = _build_final_answer_retry_conversation(base_conversation)
                    continue
                break

            needs_continue = finish_reason == 'length' or response_needs_continuation(
                answer,
                done_reason=finish_reason or '',
                expected_format=None,
            )
            if not needs_continue or continuations_used >= max_continuations:
                break

            continuations_used += 1
            conversation = _build_continuation_conversation(
                base_conversation,
                answer,
                response_format=None,
            )

        finalized = finalize_response_text(answer, expected_format=None).strip()
        if finalized != answer:
            # Patch the trailing text on the client so the assembled answer
            # matches what we'd return non-streaming.
            extra = finalized[len(answer):]
            if extra:
                yield StreamChunk(kind='token', text=extra)
            answer = finalized

        elapsed = max(1e-3, time.monotonic() - started_at)
        yield StreamChunk(
            kind='done',
            payload={
                'tokens': eval_count,
                'reasoning_tokens': reasoning_token_count,
                'prompt_tokens': prompt_eval_count,
                'elapsed_seconds': round(elapsed, 3),
                'tokens_per_second': round(eval_count / elapsed, 2) if eval_count else 0.0,
                'continuations': continuations_used,
                'answer': answer,
                'reasoning': reasoning,
            },
        )

    # -- Internal: non-streaming chat completion --------------------------

    def _complete_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        response_format: str | None,
        max_continuations: int | None = None,
    ) -> str:
        base_conversation = self._normalize_messages(messages)
        conversation = list(base_conversation)
        answer = ''
        continuations_used = 0

        if max_continuations is None:
            max_continuations = self.config.chat_continuations
        max_continuations = max(0, int(max_continuations))

        while True:
            try:
                body = self._request_chat_completion(
                    conversation,
                    model=model,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    response_format=response_format if continuations_used == 0 else None,
                )
            except Exception as exc:
                normalized = normalize_inference_error(exc, base_url=self.config.ollama_base_url)
                if normalized is not None:
                    raise normalized
                raise

            choice = (body.get('choices') or [{}])[0]
            message = choice.get('message') if isinstance(choice, dict) else None
            segment = ''
            if isinstance(message, dict):
                segment = str(message.get('content') or '')
            finish_reason = str(choice.get('finish_reason') or '').strip().lower() if isinstance(choice, dict) else ''
            if segment:
                answer = _merge_chat_segments(answer, segment)

            if not segment:
                break
            if not response_needs_continuation(answer, done_reason=finish_reason, expected_format=response_format):
                break
            if continuations_used >= max_continuations:
                break

            continuations_used += 1
            conversation = _build_continuation_conversation(
                base_conversation,
                answer,
                response_format=response_format,
            )

        return finalize_response_text(answer, expected_format=response_format).strip()

    def _request_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        response_format: str | None,
    ) -> dict[str, Any]:
        payload = self._build_request_payload(
            messages,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            response_format=response_format,
            stream=False,
        )
        request = urllib.request.Request(
            f'{self.config.ollama_base_url}/v1/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self._request_timeout_seconds(num_predict)) as response:
            return json.loads(response.read().decode('utf-8'))

    def _iter_chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        response_format: str | None,
    ) -> Iterator[dict[str, Any]]:
        payload = self._build_request_payload(
            messages,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            response_format=response_format,
            stream=True,
        )
        payload['stream_options'] = {'include_usage': True}
        request = urllib.request.Request(
            f'{self.config.ollama_base_url}/v1/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            method='POST',
        )
        timeout = self._request_timeout_seconds(num_predict)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode('utf-8', errors='replace').rstrip('\r\n')
                if not line:
                    continue
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    return
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = event.get('usage') if isinstance(event, dict) else None
                if isinstance(usage, dict):
                    yield {'type': 'usage', 'usage': usage}
                choices = event.get('choices') if isinstance(event, dict) else None
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get('delta') or {}
                    if isinstance(delta, dict):
                        # `reasoning_content` is emitted by llama-server when it
                        # is started with `--reasoning-format deepseek`. Some
                        # builds use `reasoning` as the field name instead, so we
                        # accept either.
                        reasoning = delta.get('reasoning_content')
                        if not (isinstance(reasoning, str) and reasoning):
                            reasoning = delta.get('reasoning')
                        if isinstance(reasoning, str) and reasoning:
                            yield {'type': 'reasoning', 'text': reasoning}
                        text = delta.get('content')
                        if isinstance(text, str) and text:
                            yield {'type': 'token', 'text': text}
                    finish_reason = choice.get('finish_reason')
                    if finish_reason:
                        yield {'type': 'finish', 'finish_reason': finish_reason}

    def _build_request_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        response_format: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'model': model,
            'messages': self._messages_for_openai(messages),
            'temperature': float(temperature),
            'max_tokens': int(num_predict),
            'stream': bool(stream),
            # llama-server uses n_ctx loaded at startup, but accepts these
            # OpenAI-compatible hints so we forward them. They are ignored when
            # not applicable.
            'n_ctx': int(num_ctx),
        }
        if response_format and response_format.lower() == 'json':
            payload['response_format'] = {'type': 'json_object'}
        return payload

    # -- Length / continuation heuristics (carried over from prior impl) --

    def _chat_num_predict(self, messages: list[dict[str, Any]], *, use_vision_model: bool) -> int:
        prompt, has_attachments = _latest_user_prompt(messages)
        vision_or_attachments = use_vision_model or has_attachments
        configured_max = self._chat_max_answer_tokens(use_vision_model=use_vision_model)
        budget = configured_max if vision_or_attachments else min(configured_max, 2400)

        if _requests_brief_answer(prompt):
            budget = 384
        elif _looks_like_long_form_request(prompt):
            budget = configured_max
        elif _looks_like_code_request(prompt):
            budget = max(budget, 3200)
        elif vision_or_attachments:
            budget = min(configured_max, max(budget, 1024))
        elif _requests_detailed_answer(prompt):
            budget = max(budget, 3200)
        elif _looks_like_list_or_comparison(prompt):
            budget = max(budget, 2200)
        elif _looks_like_simple_question(prompt):
            budget = 1400

        if len(prompt) > 180 and vision_or_attachments:
            budget = min(configured_max, max(budget, 1280))
        if len(prompt) > 280 and not vision_or_attachments:
            budget = max(budget, 2400)
        if (prompt.count('?') > 1 or '\n' in prompt) and vision_or_attachments:
            budget = min(configured_max, max(budget, 1280))
        if (prompt.count('?') > 1 or '\n' in prompt) and not vision_or_attachments:
            budget = max(budget, 2400)

        context_budget = _available_generation_tokens(
            messages,
            num_ctx=self._chat_num_ctx(use_vision_model=use_vision_model),
            use_vision_model=use_vision_model,
        )
        min_budget = min(_MIN_CHAT_NUM_PREDICT, configured_max)
        return max(min_budget, min(configured_max, context_budget, budget))

    def _chat_continuation_limit(self, messages: list[dict[str, Any]], *, use_vision_model: bool) -> int:
        configured_limit = min(_MAX_CHAT_CONTINUATIONS, self._chat_continuations_config(use_vision_model=use_vision_model))
        if configured_limit == 0:
            return 0
        prompt, has_attachments = _latest_user_prompt(messages)
        if _requests_brief_answer(prompt):
            return min(configured_limit, 1)
        if _looks_like_long_form_request(prompt) or _looks_like_code_request(prompt) or _requests_detailed_answer(prompt):
            return configured_limit
        if use_vision_model or has_attachments:
            return min(configured_limit, 2)
        return min(configured_limit, 2)

    def _chat_num_ctx(self, *, use_vision_model: bool) -> int:
        configured = self.config.vision_num_ctx if use_vision_model else self.config.num_ctx
        return max(512, int(configured))

    def _chat_max_answer_tokens(self, *, use_vision_model: bool) -> int:
        configured = self.config.vision_max_answer_tokens if use_vision_model else self.config.max_answer_tokens
        return max(1, int(configured))

    def _chat_continuations_config(self, *, use_vision_model: bool) -> int:
        configured = self.config.vision_chat_continuations if use_vision_model else self.config.chat_continuations
        return max(0, int(configured))

    def _request_timeout_seconds(self, num_predict: int) -> int:
        configured_timeout = max(
            _MIN_REQUEST_TIMEOUT_SECONDS,
            int(getattr(self.config, 'request_timeout_seconds', 120) or 120),
        )
        scaled_timeout = 45 + max(0, int(num_predict)) * 3 // 100
        return min(_MAX_REQUEST_TIMEOUT_SECONDS, max(configured_timeout, scaled_timeout))

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get('role') or 'user').strip().lower() or 'user'
                content = message.get('content', '')
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            parts.append(str(item['text']))
                    content = '\n'.join(part for part in parts if part)
                payload: dict[str, Any] = {'role': role, 'content': str(content or '')}
                images = message.get('images')
                if images:
                    payload['images'] = list(images)
                normalized_messages.append(payload)
                continue
            normalized_messages.append(
                {
                    'role': str(getattr(message, 'role', 'user') or 'user').strip().lower() or 'user',
                    'content': str(getattr(message, 'content', '') or ''),
                }
            )
        return normalized_messages

    @staticmethod
    def _messages_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get('role') or 'user').strip().lower() or 'user'
            text_content = str(message.get('content') or '')
            images = message.get('images') or []
            if images:
                content_parts: list[dict[str, Any]] = []
                if text_content:
                    content_parts.append({'type': 'text', 'text': text_content})
                for image in images:
                    content_parts.append(
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/jpeg;base64,{image}'},
                        }
                    )
                prepared.append({'role': role, 'content': content_parts})
            else:
                prepared.append({'role': role, 'content': text_content})
        return prepared


# ---------------------------------------------------------------------------
# Module-level helpers (continuation, prompt analysis, token estimation)
# ---------------------------------------------------------------------------


def _continuation_prompt(*, response_format: str | None) -> str:
    if str(response_format or '').lower() == 'json':
        return _JSON_CONTINUATION_PROMPT
    return _CHAT_CONTINUATION_PROMPT


def _build_continuation_conversation(
    base_conversation: list[dict[str, Any]],
    answer: str,
    *,
    response_format: str | None,
) -> list[dict[str, Any]]:
    tail = answer[-_CHAT_CONTINUATION_TAIL_CHARS:] if str(response_format or '').lower() != 'json' else answer
    return [
        *base_conversation,
        {'role': 'assistant', 'content': tail},
        {'role': 'user', 'content': _continuation_prompt(response_format=response_format)},
    ]


def _build_final_answer_retry_conversation(base_conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        *base_conversation,
        {'role': 'user', 'content': _FINAL_ANSWER_ONLY_PROMPT},
    ]


def _available_generation_tokens(
    messages: list[dict[str, Any]],
    *,
    num_ctx: int,
    use_vision_model: bool,
) -> int:
    prompt_chars = 0
    image_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        prompt_chars += len(str(message.get('role') or '')) + len(str(message.get('content') or ''))
        images = message.get('images') or []
        if isinstance(images, list):
            image_count += len(images)

    estimated_prompt_tokens = max(1, prompt_chars // 4)
    reserve_tokens = 512 + (768 if use_vision_model else 0) + image_count * 256
    available_tokens = int(num_ctx) - estimated_prompt_tokens - reserve_tokens
    return max(_MIN_CHAT_NUM_PREDICT, available_tokens)


def _latest_user_prompt(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get('role') or '').strip().lower() != 'user':
            continue

        content = str(message.get('content') or '')
        if _ATTACHMENT_CONTEXT_MARKER in content:
            content = content.split(_ATTACHMENT_CONTEXT_MARKER, maxsplit=1)[0]
        prompt = _normalize_prompt_text(content)
        has_attachments = (
            bool(message.get('images'))
            or '[attached file:' in content.lower()
            or '[attached image:' in content.lower()
        )
        return prompt, has_attachments
    return '', False


def _normalize_prompt_text(value: str) -> str:
    return re.sub(r'\s+', ' ', value).strip().lower()


def _requests_brief_answer(prompt: str) -> bool:
    return any(_contains_hint(prompt, hint) for hint in _SHORT_ANSWER_HINTS)


def _requests_detailed_answer(prompt: str) -> bool:
    return any(_contains_hint(prompt, hint) for hint in _LONG_ANSWER_HINTS)


def _looks_like_list_or_comparison(prompt: str) -> bool:
    return any(_contains_hint(prompt, hint) for hint in _LIST_OR_COMPARE_HINTS)


def _looks_like_code_request(prompt: str) -> bool:
    return any(_contains_hint(prompt, hint) for hint in _CODE_REQUEST_HINTS)


def _looks_like_long_form_request(prompt: str) -> bool:
    has_long_hint = _requests_detailed_answer(prompt) or ' long ' in f' {prompt} ' or 'very long' in prompt
    has_long_form_noun = any(_contains_hint(prompt, noun) for noun in _LONG_FORM_NOUNS)
    return has_long_hint and has_long_form_noun


def _looks_like_simple_question(prompt: str) -> bool:
    if len(prompt) > 90:
        return False
    return prompt.startswith(_SIMPLE_QUESTION_PREFIXES)


def _contains_hint(prompt: str, hint: str) -> bool:
    if not hint:
        return False
    if re.fullmatch(r'[a-z0-9_+-]+', hint):
        return re.search(rf'\b{re.escape(hint)}\b', prompt) is not None
    return hint in prompt


def _merge_chat_segments(answer: str, segment: str) -> str:
    if not answer:
        return segment
    if not segment:
        return answer

    lookback = min(len(answer), 2400)
    for prefix_length in range(min(len(segment), 400), 79, -20):
        prefix = segment[:prefix_length]
        position = answer.rfind(prefix, len(answer) - lookback)
        if position != -1:
            return answer[:position] + segment

    max_overlap = min(len(answer), len(segment), 160)
    for overlap in range(max_overlap, 0, -1):
        if answer[-overlap:] == segment[:overlap]:
            return answer + segment[overlap:]
    return answer + segment
