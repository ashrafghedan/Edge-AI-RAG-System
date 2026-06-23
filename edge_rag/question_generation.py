from __future__ import annotations

import random
import re
import uuid

from langchain_core.documents import Document

from .answering import answer_question, extractive_answer_from_documents, select_evidence_ids_for_documents
from .config import AppConfig
from .policy import LocalSecurityPolicy, get_default_security_policy
from .question_history import QuestionHistoryStore
from .retrieval import coerce_content, select_chunk_documents
from .types import GeneratedQuestion
from .utils import extract_json_object, normalize_question_text, normalize_whitespace, utc_now_iso


class QuestionGenerator:
    _SPEECH_VERB_BASE = {
        'asked': 'ask',
        'cried': 'cry',
        'replied': 'reply',
        'returned': 'return',
        'said': 'say',
        'shouted': 'shout',
        'whispered': 'whisper',
    }

    def __init__(
        self,
        config: AppConfig,
        history_store: QuestionHistoryStore,
        security_policy: LocalSecurityPolicy | None = None,
    ) -> None:
        self.config = config
        self.history_store = history_store
        self.security_policy = security_policy or get_default_security_policy()

    def generate(self, llm, answer_llm, store, dataset_id: str, session_id: str, chunks: list[Document]) -> GeneratedQuestion:
        eligible = [
            chunk for chunk in chunks
            if len(chunk.page_content.strip()) >= self.config.retrieval.min_chunk_characters_for_question
            and not self.security_policy.evaluate_text(chunk.page_content, purpose='retrieved_context').blocked
        ]
        if not eligible:
            eligible = [
                chunk for chunk in chunks
                if not self.security_policy.evaluate_text(chunk.page_content, purpose='retrieved_context').blocked
            ]
        if not eligible:
            raise ValueError('No safe text chunks are available for question generation.')

        self.history_store.prune_questions(
            lambda question: self._is_low_precision_question(question) or normalize_question_text(question) in {'', '...'},
            dataset_id=dataset_id,
        )
        recent_questions = self.history_store.recent_questions(dataset_id, limit=8)
        recent_block = '\n'.join(f'- {question}' for question in recent_questions) or '- none yet'

        deterministic = self._deterministic_question(
            dataset_id=dataset_id,
            session_id=session_id,
            eligible=eligible,
            recent_questions=recent_questions,
        )
        if deterministic is not None:
            return deterministic

        primary_attempts = min(self.config.retrieval.max_question_generation_attempts, 6)
        fallback_attempts = max(3, primary_attempts // 2)
        generation_passes = [
            (llm, 0, primary_attempts),
            (answer_llm, 0, fallback_attempts),
        ]

        for generator_llm, neighbor_window, attempts in generation_passes:
            pass_candidates: list[Document] = []
            while len(pass_candidates) < max(1, attempts):
                reshuffled = eligible[:]
                random.shuffle(reshuffled)
                pass_candidates.extend(reshuffled)
            for anchor in pass_candidates[: max(1, attempts)]:
                sample_chunks = select_chunk_documents(
                    chunks,
                    [str(anchor.metadata['chunk_id'])],
                    neighbor_window=neighbor_window,
                ) or [anchor]
                context = '\n\n'.join(
                    f"[chunk_id={chunk.metadata['chunk_id']} | source={chunk.metadata['source_name']}]\n{chunk.page_content}"
                    for chunk in sample_chunks
                )
                system_prompt = (
                    'Generate one concrete reading-comprehension question that is directly answerable from the supplied context. '
                    'Prefer a single factual detail or a single quoted action from one local passage. '
                    'Do not ask abstract attitude, belief, opinion, theme, or interpretation questions. '
                    'Do not use placeholder subjects like speaker, person, someone, or somebody. '
                    'Avoid yes/no questions. Avoid repeating earlier questions. '
                    'The question must be self-contained and must explicitly name the character, object, or event it asks about. '
                    'Do not start with unresolved pronouns like he, she, it, they, this, or that. '
                    'The model_answer must directly answer the question and must not simply restate the question wording. '
                    'Return valid JSON only with this schema: '
                    '{"question":"...","model_answer":"..."}. The model_answer must be short, precise, and grounded only in the context.'
                )
                if generator_llm is answer_llm:
                    system_prompt += ' Prefer a concrete factual question tied to a single detail from the context.'
                user_prompt = (
                    f'Previous questions to avoid:\n{recent_block}\n\n'
                    f'Context:\n{context}\n\n'
                    'Create one fresh question and its model answer.'
                )
                response = generator_llm.invoke([
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ])
                raw_text = coerce_content(response)
                try:
                    payload = extract_json_object(raw_text)
                except ValueError:
                    continue

                question = normalize_whitespace(str(payload.get('question', '')))
                model_answer = normalize_whitespace(str(payload.get('model_answer', '')))
                normalized_question = normalize_question_text(question)
                normalized_model_answer = normalize_question_text(model_answer)
                if not question or not model_answer or not normalized_question or not normalized_model_answer:
                    continue
                if len(normalized_question.split()) < 5:
                    continue
                if self._is_low_precision_question(question):
                    continue
                if self._looks_like_low_quality_answer(question, model_answer):
                    continue
                if self._question_answer_overlap(question, model_answer) >= 0.72:
                    continue
                if self._has_unresolved_pronoun_lead(question):
                    continue
                if self.security_policy.evaluate_text(
                    f'{question}\n{model_answer}',
                    purpose='generated_content',
                ).blocked:
                    continue
                if self.history_store.is_duplicate(dataset_id, question):
                    continue
                if normalize_question_text(question) in {normalize_question_text(value) for value in recent_questions}:
                    continue

                sample_chunk_ids = {str(chunk.metadata['chunk_id']) for chunk in sample_chunks}
                ordered_sample_chunk_ids = [str(chunk.metadata['chunk_id']) for chunk in sample_chunks]
                local_answer = extractive_answer_from_documents(question, sample_chunks)
                if local_answer and not self._looks_like_low_quality_answer(question, local_answer) and self._question_answer_overlap(question, local_answer) < 0.72:
                    source_names = list(dict.fromkeys(str(chunk.metadata['source_name']) for chunk in sample_chunks))
                    source_chunk_ids = select_evidence_ids_for_documents(question, local_answer, sample_chunks) or ordered_sample_chunk_ids
                    generated = GeneratedQuestion(
                        question_id=uuid.uuid4().hex[:12],
                        question=question,
                        model_answer=local_answer,
                        source_names=source_names,
                        source_chunk_ids=source_chunk_ids,
                        created_at=utc_now_iso(),
                    )
                    self.history_store.record(dataset_id, question, session_id, source_names)
                    return generated

                validation = answer_question(
                    answer_llm,
                    store,
                    self.config,
                    question,
                    source_chunks=sample_chunks,
                )
                if not validation.found:
                    continue
                if validation.evidence_ids and set(validation.evidence_ids).isdisjoint(sample_chunk_ids):
                    continue
                if self._looks_like_low_quality_answer(question, validation.answer):
                    continue
                if self._question_answer_overlap(question, validation.answer) >= 0.72:
                    continue

                source_names = validation.source_names or list(
                    dict.fromkeys(str(chunk.metadata['source_name']) for chunk in sample_chunks)
                )
                source_chunk_ids = validation.evidence_ids or [str(chunk.metadata['chunk_id']) for chunk in sample_chunks]
                generated = GeneratedQuestion(
                    question_id=uuid.uuid4().hex[:12],
                    question=question,
                    model_answer=validation.answer,
                    source_names=source_names,
                    source_chunk_ids=source_chunk_ids,
                    created_at=utc_now_iso(),
                )
                self.history_store.record(dataset_id, question, session_id, source_names)
                return generated

        raise RuntimeError('Unable to generate a new non-repeated reading-comprehension question.')

    def _deterministic_question(
        self,
        *,
        dataset_id: str,
        session_id: str,
        eligible: list[Document],
        recent_questions: list[str],
    ) -> GeneratedQuestion | None:
        shuffled = eligible[:]
        random.shuffle(shuffled)
        for chunk in shuffled:
            built = self._deterministic_speech_question(chunk)
            if built is None:
                continue
            question, model_answer = built
            normalized_question = normalize_question_text(question)
            if not normalized_question or self._is_low_precision_question(question):
                continue
            if self._looks_like_low_quality_answer(question, model_answer):
                continue
            if self.security_policy.evaluate_text(
                f'{question}\n{model_answer}',
                purpose='generated_content',
            ).blocked:
                continue
            if self.history_store.is_duplicate(dataset_id, question):
                continue
            if normalized_question in {normalize_question_text(value) for value in recent_questions}:
                continue

            source_name = str(chunk.metadata['source_name'])
            chunk_id = str(chunk.metadata['chunk_id'])
            generated = GeneratedQuestion(
                question_id=uuid.uuid4().hex[:12],
                question=question,
                model_answer=model_answer,
                source_names=[source_name],
                source_chunk_ids=[chunk_id],
                created_at=utc_now_iso(),
            )
            self.history_store.record(dataset_id, question, session_id, [source_name])
            return generated
        return None

    def _deterministic_speech_question(self, chunk: Document) -> tuple[str, str] | None:
        content = normalize_whitespace(chunk.page_content)
        if not content or not any(marker in content for marker in ('"', '“', '”')):
            return None

        cue_patterns = (
            r'(?P<verb>asked|cried|shouted|whispered)\s+'
            r'(?P<speaker>(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*'
            r'(?P<cue>[^,"\u201c\u201d]{4,80}),\s*["\u201c](?P<quote>[^"\u201d]{6,180})["\u201d]',
            r'["\u201c](?P<lead>[^"\u201d]{2,80})["\u201d],\s*'
            r'(?P<verb>asked|cried|shouted|whispered)\s+'
            r'(?P<speaker>(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*'
            r'(?P<cue>[^,"\u201c\u201d]{4,80}),\s*["\u201c](?P<quote>[^"\u201d]{6,180})["\u201d]',
        )
        for pattern in cue_patterns:
            match = re.search(pattern, content)
            if match is None:
                continue
            speaker = normalize_whitespace(match.group('speaker').strip(' ,;:-'))
            verb = match.group('verb').lower()
            quote = normalize_whitespace(match.group('quote').strip(' ,;:-'))
            cue = normalize_whitespace(match.group('cue').strip(' ,;:-'))
            if not speaker or not quote:
                continue
            contextual_phrase = self._speech_cue_phrase(cue)
            if not contextual_phrase:
                continue
            base_verb = self._SPEECH_VERB_BASE.get(verb, verb)
            question = f'What did {speaker} {base_verb} {contextual_phrase}?'
            model_answer = self._speech_model_answer(speaker, verb, quote)
            return question, model_answer

        response_patterns = (
            r'["\u201c](?P<quote>[^"\u201d]{3,120})["\u201d]\s*(?P<verb>replied|returned|said)\s+'
            r'(?P<speaker>(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:,\s*["\u201c](?P<tail>[^"\u201d]{3,180})["\u201d])?',
        )
        for pattern in response_patterns:
            match = re.search(pattern, content)
            if match is None:
                continue
            prior_quotes = re.findall(r'["\u201c]([^"\u201d]{6,180})["\u201d]', content[: match.start()])
            if not prior_quotes:
                continue
            speaker = normalize_whitespace(match.group('speaker').strip(' ,;:-'))
            verb = match.group('verb').lower()
            quote = normalize_whitespace(match.group('quote').strip(' ,;:-'))
            tail = normalize_whitespace((match.groupdict().get('tail') or '').strip(' ,;:-'))
            if tail:
                quote = normalize_whitespace(f'{quote} {tail}')
            prior_quote = self._trim_quote_for_question(prior_quotes[-1])
            if not speaker or not quote or not prior_quote:
                continue
            question = f'What did {speaker} say after hearing "{prior_quote}"?'
            model_answer = self._speech_model_answer(speaker, verb, quote)
            return question, model_answer
        return None

    @staticmethod
    def _speech_model_answer(speaker: str, verb: str, quote: str) -> str:
        if verb == 'asked':
            return f'{speaker} asked, "{quote}"'
        if verb == 'whispered':
            return f'{speaker} whispered, "{quote}"'
        if verb == 'replied':
            return f'{speaker} replied, "{quote}"'
        if verb == 'returned':
            return f'{speaker} returned, "{quote}"'
        if verb == 'cried':
            return f'{speaker} cried, "{quote}"'
        if verb == 'shouted':
            return f'{speaker} shouted, "{quote}"'
        return f'{speaker} said, "{quote}"'

    @staticmethod
    def _speech_cue_phrase(cue: str) -> str:
        normalized = normalize_whitespace(cue)
        if not normalized:
            return ''
        lowered = normalized.lower()
        words = normalize_question_text(normalized).split()
        if len(words) < 2:
            return ''
        if words[0] in {'while', 'when', 'after', 'before', 'during', 'upon', 'as'}:
            return lowered
        if words[0] in {'with', 'without', 'in', 'on', 'at', 'under'}:
            return f'while {lowered}'
        if any(word.endswith('ing') for word in words):
            return f'while {lowered}'
        return ''

    @staticmethod
    def _trim_quote_for_question(quote: str) -> str:
        normalized = normalize_whitespace(quote)
        if not normalized:
            return ''
        words = normalized.split()
        if len(words) <= 14 and len(normalized) <= 90:
            return normalized
        trimmed = ' '.join(words[:14]).rstrip(' ,;:-')
        return trimmed + '...'
    @staticmethod
    def _question_answer_overlap(question: str, answer: str) -> float:
        question_tokens = set(normalize_question_text(question).split())
        answer_tokens = set(normalize_question_text(answer).split())
        if not question_tokens or not answer_tokens:
            return 0.0
        return len(question_tokens & answer_tokens) / max(len(question_tokens | answer_tokens), 1)

    @staticmethod
    def _has_unresolved_pronoun_lead(question: str) -> bool:
        lowered = question.lower().strip()
        return lowered.startswith(
            (
                'what does he ',
                'what did he ',
                'what does she ',
                'what did she ',
                'what does it ',
                'what did it ',
                'why does he ',
                'why did he ',
                'why does she ',
                'why did she ',
                'why does it ',
                'why did it ',
                'how does he ',
                'how did he ',
                'how does she ',
                'how did she ',
                'how does it ',
                'how did it ',
                'what does this ',
                'what does that ',
                'what causes it ',
                'what change is noted ',
            )
        )

    @staticmethod
    def _is_low_precision_question(question: str) -> bool:
        lowered = normalize_question_text(question)
        if lowered in {'', '...'}:
            return True
        if re.fullmatch(
            r'what did (?:(?:mr|mrs|ms|miss|dr) )?[a-z]+(?: [a-z]+)? (?:say|ask|reply|return|cry|shout|whisper)',
            lowered,
        ):
            return True
        if any(token in f' {lowered} ' for token in (' speaker ', ' someone ', ' somebody ', ' person ')):
            return True
        if lowered.startswith('what did the ') and any(
            token in f' {lowered} ' for token in (' bundle ', ' thing ', ' object ', ' item ')
        ):
            return True
        if lowered.endswith((' to', ' for', ' with', ' about')):
            return True
        return any(
            phrase in lowered
            for phrase in (
                'believe about',
                'think about',
                'attitude regarding',
                'attitude toward',
                'opinion of',
                'feel about',
                'what happens',
                'what happened',
                'what change is noted',
            )
        )

    @staticmethod
    def _looks_like_low_quality_answer(question: str, answer: str) -> bool:
        normalized_answer = normalize_question_text(answer)
        if not normalized_answer or normalized_answer == '...':
            return True
        if len(normalized_answer.split()) < 4:
            return True
        if len(answer) > 200:
            return True
        sentence_like = re.sub(r'\b(Mr|Mrs|Ms|Dr|Miss)\.', r'\1', answer, flags=re.IGNORECASE)
        if sum(sentence_like.count(marker) for marker in '.?!') > 1:
            return True
        if answer.lower().startswith(('and ', 'but ', 'or ', 'so ', 'then ')):
            return True
        if normalize_question_text(question) == normalized_answer:
            return True
        return False


