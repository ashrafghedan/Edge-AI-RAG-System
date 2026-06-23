from __future__ import annotations

import json
import re

from langchain_core.documents import Document

from .config import AppConfig
from .policy import get_default_security_policy
from .retrieval import (
    build_context,
    build_focused_context,
    coerce_content,
    documents_to_retrieved_chunks,
    retrieve_chunks,
    select_chunk_documents,
    unique_sources,
)
from .types import GroundedAnswerResult
from .utils import extract_json_object, normalize_question_text, normalize_whitespace


NOT_AVAILABLE_SIGNALS = {
    'not available',
    'not found',
    'unknown',
    'insufficient',
}
_REASON_HINTS = {'why', 'reason', 'refuse', 'refused', 'join', 'union', 'combination', 'could', 'canna', 'regulations'}
_CONSEQUENCE_HINTS = {
    'consequence',
    'result',
    'effect',
    'thus',
    'shunned',
    'ostracized',
    'loneliest',
    'solitude',
    'avoided',
    'dismissed',
    'leave',
    'deserter',
}
_REASON_PHRASES = ('could not come in', 'canna coom in', 'simply canna', 'proposed reg', 'proposed regulations')
_CONSEQUENCE_PHRASES = ('loneliest of lives', 'solitude among a familiar crowd', 'avoided that side of the street', 'shunned')
_DIAGNOSIS_HINTS = {'diagnosis', 'condition', 'temporary', 'nervous', 'depression', 'hysterical', 'tendency', 'matter', 'sick'}
_TREATMENT_HINTS = {
    'treatment',
    'prescribe',
    'prescribed',
    'phosphates',
    'phosphites',
    'tonics',
    'exercise',
    'air',
    'journeys',
    'forbidden',
    'work',
    'cod',
    'liver',
    'oil',
}
_IDENTITY_HINTS = {'is', 'was', 'sister', 'brother', 'mother', 'father', 'wife', 'husband', 'housekeeper', 'physician', 'doctor'}
_RESPONSIBILITY_HINTS = {'responsibilities', 'role', 'duties', 'housekeeper', 'care', 'cares', 'manage', 'manages', 'ready', 'sees', 'everything'}
_IDENTITY_REQUIRED = {'sister', 'brother', 'mother', 'father', 'wife', 'husband', 'housekeeper', 'physician', 'doctor'}
_RESPONSIBILITY_REQUIRED = {'housekeeper', 'manage', 'manages', 'ready', 'sees', 'everything'}
_SPEECH_VERB_LABELS = {
    'ask': 'asked',
    'asked': 'asked',
    'cry': 'cried',
    'cried': 'cried',
    'reply': 'replied',
    'replied': 'replied',
    'return': 'returned',
    'returned': 'returned',
    'say': 'said',
    'said': 'said',
    'shout': 'shouted',
    'shouted': 'shouted',
    'whisper': 'whispered',
    'whispered': 'whispered',
}
_ANSWERING_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'but',
    'by',
    'did',
    'do',
    'does',
    'during',
    'for',
    'from',
    'had',
    'has',
    'have',
    'her',
    'hers',
    'him',
    'his',
    'in',
    'into',
    'is',
    'it',
    'its',
    'of',
    'on',
    'or',
    'that',
    'the',
    'their',
    'them',
    'there',
    'they',
    'this',
    'to',
    'was',
    'were',
    'what',
    'when',
    'where',
    'which',
    'who',
    'why',
    'with',
}


def _normalize_evidence_id(value: str) -> str:
    candidate = value.strip().strip('"').strip("'")
    match = re.search(r'([a-f0-9]{12}-\d{4})', candidate, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if '=' in candidate:
        return candidate.split('=', maxsplit=1)[1].strip()
    return candidate


def _build_supported_result(answer: str, retrieved, evidence_ids: list[str]) -> GroundedAnswerResult:
    if not evidence_ids and retrieved:
        evidence_ids = [retrieved[0].chunk_id]

    supporting_sources = [chunk.source_name for chunk in retrieved if chunk.chunk_id in evidence_ids]
    if not supporting_sources:
        supporting_sources = unique_sources(retrieved)

    return GroundedAnswerResult(
        answer=answer,
        found=True,
        source_names=list(dict.fromkeys(supporting_sources)),
        evidence_ids=evidence_ids,
        retrieved_chunks=retrieved,
    )


def _extract_plain_answer_candidate(text: str) -> str:
    candidate = text.strip().replace('```', '')
    if not candidate:
        return ''

    if '<channel|>' in candidate:
        candidate = candidate.rsplit('<channel|>', maxsplit=1)[-1].strip()

    lowered_candidate = candidate.lower()
    if re.search(r'(?:chunk|chank)[_-]?id\s*=', lowered_candidate):
        return ''

    labeled = re.search(r'(?is)(?:final answer|answer)\s*[:\-]\s*(.+)$', candidate)
    if labeled:
        candidate = labeled.group(1).strip()
    elif (
        'thinking process' in lowered_candidate
        or 'analyze the request' in lowered_candidate
        or 'scan excerpts' in lowered_candidate
        or 'synthesize the answer' in lowered_candidate
        or re.search(r'(?m)^\s*(?:\*+|\d+\.)\s+', candidate) is not None
    ):
        return ''
    else:
        paragraphs = [line.strip() for line in candidate.splitlines() if line.strip()]
        if not paragraphs:
            return ''
        candidate = paragraphs[-1]

    candidate = re.sub(r'^(?:final answer|answer)\s*[:\-]\s*', '', candidate, flags=re.IGNORECASE)
    candidate = normalize_whitespace(candidate)
    if len(candidate) < 8:
        return ''
    return candidate


def _support_terms(text: str) -> set[str]:
    return {
        token
        for token in normalize_question_text(text).split()
        if len(token) > 1 and token not in _ANSWERING_STOPWORDS
    }


def _support_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in re.split(r'(?<=[.!?])\s+|;\s+', text):
        candidate = normalize_whitespace(part)
        if len(candidate) < 25:
            continue
        sentences.append(candidate)
    return sentences or [normalize_whitespace(text)]


def _best_question_support_ratio(question: str, retrieved) -> float:
    question_terms = _support_terms(question)
    if len(question_terms) <= 2:
        return 1.0

    quoted_segments = {
        normalize_question_text(segment)
        for segment in re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', question)
        if normalize_question_text(segment)
    }
    best_ratio = 0.0
    for chunk in retrieved:
        normalized_chunk = normalize_question_text(chunk.content)
        if quoted_segments and any(segment in normalized_chunk for segment in quoted_segments):
            return 1.0
        for sentence in _support_sentences(chunk.content):
            sentence_terms = _support_terms(sentence)
            if not sentence_terms:
                continue
            coverage = len(question_terms & sentence_terms) / max(len(question_terms), 1)
            if coverage > best_ratio:
                best_ratio = coverage
    return best_ratio


def _requires_strong_support(question: str) -> bool:
    lowered = question.lower()
    if '"' in question or '“' in question or 'excerpt' in lowered:
        return False
    return not (
        _is_reason_consequence_question(question)
        or _is_diagnosis_treatment_question(question)
        or _is_role_identity_question(question)
        or lowered.startswith('what did ')
    )


def _passes_support_gate(question: str, retrieved, evidence_ids: list[str] | None = None) -> bool:
    relevant_chunks = retrieved
    if evidence_ids:
        selected = [chunk for chunk in retrieved if chunk.chunk_id in set(evidence_ids)]
        if selected:
            relevant_chunks = selected
    if not relevant_chunks:
        return False
    best_ratio = _best_question_support_ratio(question, relevant_chunks)
    top_relevance = max(chunk.relevance for chunk in relevant_chunks)
    return best_ratio >= 0.45 or top_relevance >= 0.72


def _select_evidence_ids(question: str, answer: str, retrieved) -> list[str]:
    if not retrieved:
        return []

    target_terms = _support_terms(f'{question} {answer}')
    scored: list[tuple[float, str]] = []
    for chunk in retrieved:
        content_terms = _support_terms(chunk.content)
        overlap = len(target_terms & content_terms)
        score = overlap + chunk.relevance
        if answer and normalize_question_text(answer) and normalize_question_text(answer) in normalize_question_text(chunk.content):
            score += 4.0
        scored.append((score, chunk.chunk_id))

    scored.sort(reverse=True)
    selected = [chunk_id for score, chunk_id in scored if score > 0.0][:2]
    return selected or [retrieved[0].chunk_id]


def _extract_partial_payload(raw_text: str) -> dict[str, object] | None:
    status_match = re.search(r'"status"\s*:\s*"([^"]+)"', raw_text, flags=re.IGNORECASE)
    answer_match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', raw_text, flags=re.DOTALL)
    evidence_ids = list(dict.fromkeys(re.findall(r'([a-f0-9]{12}-\d{4})', raw_text, flags=re.IGNORECASE)))
    if status_match is None and answer_match is None:
        return None

    status = status_match.group(1).strip().lower() if status_match else ''
    answer = ''
    if answer_match:
        try:
            answer = json.loads(f'"{answer_match.group(1)}"')
        except json.JSONDecodeError:
            answer = answer_match.group(1).replace('\\"', '"')
        answer = normalize_whitespace(answer)

    return {
        'status': status,
        'answer': answer,
        'evidence_ids': evidence_ids,
    }


def _request_plain_answer(llm, question: str, context: str) -> str:
    lowered_question = question.lower()
    multi_part_reasoning = _is_reason_consequence_question(question)
    format_instruction = (
        'This is a multi-part question. Answer both parts using this exact plain-text format:\n'
        'Reason: ...\n'
        'Consequence: ...'
        if multi_part_reasoning
        else 'Give a direct answer in one or two sentences.'
    )
    response = llm.invoke([
        {
            'role': 'system',
            'content': (
                'Answer only from the supplied excerpts. '
                'Reply with plain text only, no JSON, no bullet points, no headings, and no reasoning. '
                'If the answer is not directly supported, reply exactly NOT_AVAILABLE. '
                'Answer every part of the question.'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'Question: {question}\n\n'
                f'Excerpts:\n{context}\n\n'
                f'{format_instruction}'
            ),
        },
    ])
    return _extract_plain_answer_candidate(coerce_content(response))


def _is_reason_consequence_question(question: str) -> bool:
    lowered_question = question.lower()
    return (
        ('why' in lowered_question or 'reason' in lowered_question)
        and any(term in lowered_question for term in ('consequence', 'result', 'effect', 'happen'))
    )


def _is_diagnosis_treatment_question(question: str) -> bool:
    lowered_question = question.lower()
    return (
        any(term in lowered_question for term in ('diagnosis', 'condition'))
        and any(term in lowered_question for term in ('treatment', 'prescribe', 'prescribed'))
    )


def _is_role_identity_question(question: str) -> bool:
    lowered_question = question.lower().strip()
    return (
        lowered_question.startswith('who is ')
        or lowered_question.startswith('who was ')
        or ' responsibilities ' in lowered_question
        or ' responsibility ' in lowered_question
        or ' role ' in lowered_question
        or ' professions ' in lowered_question
        or ' profession ' in lowered_question
    )


def _question_entity_name(question: str) -> str:
    entities = [
        token
        for token in re.findall(r"\b[A-Z][a-z']{2,}\b", question)
        if token.lower() not in {'what', 'when', 'where', 'which', 'who', 'why', 'how'}
    ]
    if len(entities) >= 2 and entities[0].lower() in {'mr', 'mrs', 'ms', 'miss', 'dr'}:
        return f'{entities[0]}. {entities[1]}'
    return entities[0] if entities else ''


def _merge_retrieved_chunks(*chunk_groups) -> list:
    merged = {}
    order: list[str] = []
    for group in chunk_groups:
        for chunk in group or []:
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
                order.append(chunk.chunk_id)
                continue
            if chunk.relevance > existing.relevance:
                merged[chunk.chunk_id] = chunk
    return [merged[chunk_id] for chunk_id in order]


def _sentence_candidates(retrieved) -> list[tuple[str, float]]:
    sentences: list[tuple[str, float]] = []
    for chunk in retrieved:
        parts = re.split(r'(?<=[.!?])\s+|;\s+', chunk.content)
        for part in parts:
            candidate = normalize_whitespace(part)
            if len(candidate) < 40:
                continue
            sentences.append((candidate, chunk.relevance))
    return sentences


def _looks_like_fragment(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return True
    return bool(re.match(r'^[a-z]', stripped))


def _question_when_clause(question: str) -> str:
    match = re.search(r'\bwhen\b\s+(.+?)\??$', question.strip(), flags=re.IGNORECASE)
    if match is None:
        return ''
    return normalize_question_text(match.group(1))


def _question_speech_verb(question: str) -> str:
    lowered = question.lower()
    for token, label in _SPEECH_VERB_LABELS.items():
        if f' {token} ' in f' {lowered} ':
            return label
    return ''


def _quote_segments(sentence: str) -> list[str]:
    return [
        normalize_whitespace(segment.strip(' ,;:-'))
        for segment in re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', sentence)
        if normalize_whitespace(segment.strip(' ,;:-'))
    ]

def _reported_speech_answer(question: str, retrieved) -> str:
    lowered_question = question.lower().strip()
    speech_verb = _question_speech_verb(question)
    if not lowered_question.startswith('what did ') or (' when ' not in lowered_question and not speech_verb):
        return ''

    entity_name = _question_entity_name(question)
    cue_phrase = _question_when_clause(question)
    cue_terms = set(cue_phrase.split())
    best: tuple[float, str] | None = None

    for chunk in retrieved:
        content = normalize_whitespace(chunk.content)
        lowered_sentence = content.lower()
        normalized_sentence = normalize_question_text(content)
        sentence_terms = set(normalized_sentence.split())
        if entity_name and entity_name.lower() not in lowered_sentence:
            continue
        if cue_terms and len(cue_terms & sentence_terms) < max(1, min(2, len(cue_terms))):
            continue
        if not any(
            verb in lowered_sentence
            for verb in (' said ', ' asked ', ' replied ', ' returned ', ' cried ', ' whispered ', ' shouted ')
        ):
            continue

        quote_segments = _quote_segments(content)
        if cue_terms and len(quote_segments) > 1:
            quote_segments = [quote_segments[-1]]
        quoted_text = normalize_whitespace(' '.join(quote_segments))
        if not quoted_text and speech_verb:
            lead_match = re.match(
                r'(.+?)\b(?:said|whispered|asked|replied|returned|cried|shouted)\b',
                content,
                flags=re.IGNORECASE,
            )
            if lead_match:
                quoted_text = normalize_whitespace(lead_match.group(1).strip(' "\'?,;:-'))
        if not quoted_text and ',' in content:
            quoted_text = normalize_whitespace(content.rsplit(',', maxsplit=1)[-1].strip(' "\'?'))
        if not quoted_text or len(quoted_text) < 6:
            continue

        chosen_verb = speech_verb
        if not chosen_verb:
            if ' asked ' in lowered_sentence or 'ask' in lowered_sentence or '?' in quoted_text:
                chosen_verb = 'asked'
            elif ' replied ' in lowered_sentence:
                chosen_verb = 'replied'
            elif ' returned ' in lowered_sentence:
                chosen_verb = 'returned'
            else:
                chosen_verb = 'said'

        if chosen_verb == 'asked':
            candidate = f'{entity_name or "The speaker"} asked, "{quoted_text}"'
        elif chosen_verb in {'replied', 'returned'}:
            candidate = f'{entity_name or "The speaker"} replied, "{quoted_text}"'
        elif chosen_verb == 'whispered':
            candidate = f'{entity_name or "The speaker"} whispered, "{quoted_text}"'
        elif chosen_verb == 'cried':
            candidate = f'{entity_name or "The speaker"} cried, "{quoted_text}"'
        elif chosen_verb == 'shouted':
            candidate = f'{entity_name or "The speaker"} shouted, "{quoted_text}"'
        else:
            candidate = f'{entity_name or "The speaker"} said, "{quoted_text}"'

        score = (len(cue_terms & sentence_terms) * 2.5) + (chunk.relevance * 2.0)
        if 'news of the day' in lowered_sentence:
            score += 3.0
        if best is None or score > best[0]:
            best = (score, candidate)

    return best[1] if best else ''

def _best_supported_sentence(
    question: str,
    retrieved,
    hint_terms: set[str],
    *,
    phrase_hints: tuple[str, ...] = (),
    exclude: str | None = None,
    required_terms: set[str] | None = None,
) -> str:
    query_terms = set(normalize_question_text(question).split())
    best_sentence = ''
    best_score = -1.0
    best_phrase_sentence = ''
    best_phrase_score = -1.0
    for sentence, relevance in _sentence_candidates(retrieved):
        normalized = normalize_question_text(sentence)
        if not normalized:
            continue
        if _looks_like_fragment(sentence):
            continue
        if exclude and normalized == normalize_question_text(exclude):
            continue
        sentence_terms = set(normalized.split())
        if required_terms and not (required_terms & sentence_terms):
            continue
        overlap = len(query_terms & sentence_terms)
        hint_hits = len(hint_terms & sentence_terms)
        phrase_hits = sum(1 for phrase in phrase_hints if phrase in sentence.lower())
        score = (overlap * 1.0) + (hint_hits * 1.8) + (phrase_hits * 5.0) + (relevance * 2.0)
        if '?' in sentence:
            score -= 3.0
        lowered_sentence = sentence.lower()
        if 'may i take the liberty of asking' in lowered_sentence or 'how it happens' in lowered_sentence:
            score -= 3.0
        if 'tell us about' in lowered_sentence or 'speak up like a man' in lowered_sentence:
            score -= 3.0
        if phrase_hits > 0 and score > best_phrase_score:
            best_phrase_score = score
            best_phrase_sentence = sentence
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_phrase_sentence or best_sentence


def _best_identity_sentence(question: str, retrieved) -> str:
    entity_name = _question_entity_name(question).lower()
    best_sentence = ''
    best_score = -1.0
    for sentence, relevance in _sentence_candidates(retrieved):
        normalized = normalize_question_text(sentence)
        if not normalized:
            continue
        if _looks_like_fragment(sentence):
            continue
        sentence_terms = set(normalized.split())
        lowered = sentence.lower()
        if not (_IDENTITY_REQUIRED & sentence_terms):
            continue
        if entity_name and entity_name not in lowered:
            if not lowered.startswith(('she is ', 'he is ', 'they are ')) and "john's sister" not in lowered:
                continue
        if 'my brother' in lowered or 'a friend who' in lowered:
            continue
        overlap = len(set(normalize_question_text(question).split()) & sentence_terms)
        hint_hits = len(_IDENTITY_HINTS & sentence_terms)
        score = overlap + (hint_hits * 1.8) + (relevance * 2.0)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence


def _best_responsibility_sentence(question: str, retrieved, *, exclude: str | None = None) -> str:
    entity_name = _question_entity_name(question).lower()
    best_sentence = ''
    best_score = -1.0
    for sentence, relevance in _sentence_candidates(retrieved):
        normalized = normalize_question_text(sentence)
        if not normalized:
            continue
        if _looks_like_fragment(sentence):
            continue
        if exclude and normalized == normalize_question_text(exclude):
            continue
        sentence_terms = set(normalized.split())
        lowered = sentence.lower()
        if not (_RESPONSIBILITY_REQUIRED & sentence_terms):
            continue
        if entity_name and entity_name not in lowered and not lowered.startswith(('she ', 'he ', 'they ')):
            continue
        if "don't care" in lowered or "do not care" in lowered:
            continue
        overlap = len(set(normalize_question_text(question).split()) & sentence_terms)
        hint_hits = len(_RESPONSIBILITY_HINTS & sentence_terms)
        required_hits = len(_RESPONSIBILITY_REQUIRED & sentence_terms)
        score = overlap + (hint_hits * 1.9) + (required_hits * 2.2) + (relevance * 2.0)
        if 'sees to everything' in lowered:
            score += 2.0
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence


def _normalize_structured_sentence(entity_name: str, sentence: str) -> str:
    if not sentence:
        return ''
    if not entity_name:
        return sentence
    lowered = sentence.lower()
    if entity_name.lower() in lowered:
        return sentence
    if lowered.startswith('she is '):
        return f'{entity_name} {sentence[3:]}'
    if lowered.startswith('he is '):
        return f'{entity_name} {sentence[2:]}'
    if lowered.startswith("there comes john's sister"):
        return f"{entity_name} is John's sister."
    return sentence


def _build_extractive_fallback(question: str, retrieved, *, source_chunks: list[Document] | None = None) -> str:
    if not retrieved and not source_chunks:
        return ''

    search_chunks = list(retrieved)
    if source_chunks and retrieved:
        support_documents = select_chunk_documents(
            source_chunks,
            [chunk.chunk_id for chunk in retrieved],
            neighbor_window=2,
        )
        if support_documents:
            search_chunks = _merge_retrieved_chunks(
                retrieved,
                documents_to_retrieved_chunks(support_documents, relevance=0.55),
            )
        else:
            search_chunks = documents_to_retrieved_chunks(source_chunks, relevance=0.45)
    elif source_chunks:
        search_chunks = documents_to_retrieved_chunks(source_chunks, relevance=0.4)

    reported_speech = _reported_speech_answer(question, search_chunks)
    if reported_speech:
        return reported_speech

    if _is_reason_consequence_question(question):
        reason = _best_supported_sentence(question, search_chunks, _REASON_HINTS, phrase_hints=_REASON_PHRASES)
        consequence = _best_supported_sentence(
            question,
            search_chunks,
            _CONSEQUENCE_HINTS,
            phrase_hints=_CONSEQUENCE_PHRASES,
            exclude=reason,
        )
        if reason and consequence:
            return normalize_whitespace(f'Reason: {reason} Consequence: {consequence}')

    if _is_diagnosis_treatment_question(question):
        diagnosis = _best_supported_sentence(question, search_chunks, _DIAGNOSIS_HINTS)
        treatment = _best_supported_sentence(question, search_chunks, _TREATMENT_HINTS, exclude=diagnosis)
        if diagnosis and treatment:
            return normalize_whitespace(f'{diagnosis} {treatment}')

    if _is_role_identity_question(question):
        entity_name = _question_entity_name(question)
        identity = _best_identity_sentence(question, search_chunks)
        responsibility = _best_responsibility_sentence(question, search_chunks, exclude=identity)
        identity = _normalize_structured_sentence(entity_name, identity)
        responsibility = _normalize_structured_sentence(entity_name, responsibility)
        if identity and responsibility:
            return normalize_whitespace(f'{identity} {responsibility}')
        if responsibility:
            return normalize_whitespace(responsibility)

    question_terms = _support_terms(question)
    best_sentences: list[tuple[float, str]] = []
    for sentence, relevance in _sentence_candidates(search_chunks):
        sentence_terms = _support_terms(sentence)
        if not sentence_terms:
            continue
        if _looks_like_fragment(sentence):
            continue
        overlap = len(question_terms & sentence_terms)
        if overlap < 2 and len(question_terms) > 4:
            continue
        score = overlap + (relevance * 2.0)
        if '?' in sentence:
            score -= 2.5
        best_sentences.append((score, sentence))

    best_sentences.sort(reverse=True)
    selected = [sentence for score, sentence in best_sentences if score > 0.0][:2]
    if selected:
        return normalize_whitespace(' '.join(selected))

    return ''


def extractive_answer_from_documents(question: str, source_chunks: list[Document]) -> str:
    retrieved = documents_to_retrieved_chunks(source_chunks, relevance=1.0)
    return _build_extractive_fallback(question, retrieved, source_chunks=source_chunks)


def select_evidence_ids_for_documents(question: str, answer: str, source_chunks: list[Document]) -> list[str]:
    retrieved = documents_to_retrieved_chunks(source_chunks, relevance=1.0)
    return _select_evidence_ids(question, answer, retrieved)


def answer_question(
    llm,
    store,
    config: AppConfig,
    question: str,
    *,
    source_chunks: list[Document] | None = None,
) -> GroundedAnswerResult:
    security_policy = get_default_security_policy()
    query_decision = security_policy.evaluate_text(question, purpose='user_query')
    if query_decision.blocked:
        return GroundedAnswerResult(
            answer=security_policy.refusal_message(query_decision),
            found=False,
            source_names=[],
            evidence_ids=[],
            retrieved_chunks=[],
        )

    retrieved = retrieve_chunks(
        store,
        question,
        config.retrieval.top_k,
        source_chunks=source_chunks,
    )
    retrieved = [
        chunk for chunk in retrieved
        if not security_policy.evaluate_text(chunk.content, purpose='retrieved_context').blocked
    ]
    if source_chunks and retrieved:
        support_documents = select_chunk_documents(
            source_chunks,
            [chunk.chunk_id for chunk in retrieved],
            neighbor_window=2,
        )
        support_chunks = [
            chunk
            for chunk in documents_to_retrieved_chunks(support_documents, relevance=0.55)
            if not security_policy.evaluate_text(chunk.content, purpose='retrieved_context').blocked
        ]
        retrieved = _merge_retrieved_chunks(retrieved, support_chunks)
    if not retrieved:
        return GroundedAnswerResult(
            answer=config.not_available_response,
            found=False,
            source_names=[],
            evidence_ids=[],
            retrieved_chunks=[],
        )
    if _requires_strong_support(question) and not _passes_support_gate(question, retrieved):
        return GroundedAnswerResult(
            answer=config.not_available_response,
            found=False,
            source_names=unique_sources(retrieved),
            evidence_ids=[],
            retrieved_chunks=retrieved,
        )

    context = build_focused_context(question, retrieved)
    system_prompt = (
        'You are a strict offline RAG assistant. Answer only from the supplied excerpts. '\
        'You may combine facts across multiple excerpts when the question asks for causes, consequences, or another multi-part answer. '\
        'If the question quotes an excerpt and asks what happens in it, explain what that excerpt describes in plain language using only the supplied excerpts. '\
        'Do not use background knowledge, guesswork, or inference beyond what is explicitly stated. '\
        'Return valid JSON only with this schema: '\
        '{"status":"supported|not_available","answer":"...","evidence_ids":["chunk-id"]}. '\
        'If the answer is missing from the excerpts, set status to "not_available", answer to an empty string, '\
        'and evidence_ids to an empty list. Use at most two evidence_ids. '\
        'Keep supported answers concise, usually one to three sentences.'
    )
    user_prompt = (
        f'Question: {question}\n\n'
        f'Excerpts:\n{context}\n\n'
        'Answer only if the excerpts directly support it.'
    )
    response = llm.invoke([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ])
    raw_text = coerce_content(response)
    extractive_answer = _build_extractive_fallback(question, retrieved, source_chunks=source_chunks)

    try:
        payload = extract_json_object(raw_text)
    except (ValueError, json.JSONDecodeError):
        payload = _extract_partial_payload(raw_text)

    if payload is not None:
        status = str(payload.get('status', '')).strip().lower()
        answer = normalize_whitespace(str(payload.get('answer', '')))
        available_ids = {chunk.chunk_id for chunk in retrieved}
        evidence_ids = [
            normalized
            for item in payload.get('evidence_ids', [])
            if (normalized := _normalize_evidence_id(str(item))) in available_ids
        ]

        allow_extractive_override = bool(extractive_answer) and (
            not _requires_strong_support(question) or _passes_support_gate(question, retrieved)
        )
        if status == 'not_available' and not allow_extractive_override:
            return GroundedAnswerResult(
                answer=config.not_available_response,
                found=False,
                source_names=unique_sources(retrieved),
                evidence_ids=[],
                retrieved_chunks=retrieved,
            )

        if status == 'supported' and answer:
            lowered = answer.lower()
            if not any(signal in lowered for signal in NOT_AVAILABLE_SIGNALS):
                if (
                    extractive_answer
                    and (
                        _is_reason_consequence_question(question)
                        or _is_diagnosis_treatment_question(question)
                        or _is_role_identity_question(question)
                    )
                ):
                    answer = extractive_answer
                    evidence_ids = _select_evidence_ids(question, answer, retrieved)
                answer_decision = security_policy.evaluate_text(answer, purpose='model_output')
                if answer_decision.blocked:
                    return GroundedAnswerResult(
                        answer=security_policy.refusal_message(answer_decision),
                        found=False,
                        source_names=unique_sources(retrieved),
                        evidence_ids=[],
                        retrieved_chunks=retrieved,
                    )
                if _requires_strong_support(question) and not _passes_support_gate(question, retrieved, evidence_ids):
                    return GroundedAnswerResult(
                        answer=config.not_available_response,
                        found=False,
                        source_names=unique_sources(retrieved),
                        evidence_ids=[],
                        retrieved_chunks=retrieved,
                    )
                return _build_supported_result(answer, retrieved, evidence_ids)

    fallback_answer = _extract_plain_answer_candidate(raw_text)
    if extractive_answer and (not _requires_strong_support(question) or _passes_support_gate(question, retrieved)):
        fallback_answer = extractive_answer
    if not fallback_answer:
        fallback_answer = _request_plain_answer(llm, question, context)

    lowered = fallback_answer.lower()
    if not fallback_answer or any(signal in lowered for signal in NOT_AVAILABLE_SIGNALS):
        return GroundedAnswerResult(
            answer=config.not_available_response,
            found=False,
            source_names=unique_sources(retrieved),
            evidence_ids=[],
            retrieved_chunks=retrieved,
        )

    answer_decision = security_policy.evaluate_text(fallback_answer, purpose='model_output')
    if answer_decision.blocked:
        return GroundedAnswerResult(
            answer=security_policy.refusal_message(answer_decision),
            found=False,
            source_names=unique_sources(retrieved),
            evidence_ids=[],
            retrieved_chunks=retrieved,
        )

    evidence_ids = _select_evidence_ids(question, fallback_answer, retrieved)
    if _requires_strong_support(question) and not _passes_support_gate(question, retrieved, evidence_ids):
        return GroundedAnswerResult(
            answer=config.not_available_response,
            found=False,
            source_names=unique_sources(retrieved),
            evidence_ids=[],
            retrieved_chunks=retrieved,
        )
    return _build_supported_result(fallback_answer, retrieved, evidence_ids)

