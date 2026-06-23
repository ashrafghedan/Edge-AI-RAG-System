from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher

from langchain_core.documents import Document

from .config import AppConfig
from .retrieval import build_context, coerce_content, documents_to_retrieved_chunks, retrieve_chunks, select_chunk_documents
from .types import GeneratedQuestion, GradingResult
from .utils import extract_json_object, normalize_question_text, normalize_whitespace


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


_RUBRIC_FEEDBACK = {
    10: 'Completely correct, precise, and fully supported by the text.',
    9: 'Correct and nearly complete, with only a tiny wording or detail gap.',
    8: 'Clearly correct on the core answer, with only a small omission or imprecision.',
    7: 'Mostly correct, but one notable detail is missing or slightly inaccurate.',
    6: 'More right than wrong; the main idea is present, but a major detail is missing or underspecified.',
    5: 'Partially correct; the answer shows relevant understanding, but important parts are missing or unclear.',
    4: 'Limited partial answer; it is on topic and somewhat supported, but it misses the main asked detail.',
    3: 'Only slight overlap with the source-backed answer; most of the answer is incomplete or off-target.',
    2: 'Almost entirely incorrect, though it is still loosely on topic.',
    1: 'Incorrect and unsupported by the relevant source text.',
    0: 'No usable answer or an answer that is irrelevant to the question.',
}

_GRADING_STOPWORDS = {
    'a',
    'an',
    'and',
    'appeared',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'he',
    'her',
    'hers',
    'him',
    'his',
    'in',
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
    'these',
    'this',
    'those',
    'to',
    'they',
    'said',
    'say',
    'replied',
    'reply',
    'returned',
    'return',
    'she',
    'asked',
    'ask',
    'whispered',
    'whisper',
    'shouted',
    'shout',
    'cried',
    'cry',
    'mr',
    'mrs',
    'ms',
    'miss',
    'dr',
    'was',
    'were',
    'with',
}

_CANONICAL_TOKEN_OVERRIDES = {
    'maybe': 'perhaps',
    'wi': 'with',
}
_DIALOGUE_VERBS = {'say', 'reply', 'return', 'ask', 'whisper', 'shout', 'cry'}
_MANNER_ONLY_TERMS = {
    'acknowledge',
    'acknowledging',
    'attentive',
    'attentively',
    'calm',
    'calmly',
    'continued',
    'dismissive',
    'dismissively',
    'firm',
    'firmly',
    'listen',
    'listened',
    'manner',
    'nod',
    'nodded',
    'polite',
    'politely',
    'quiet',
    'quietly',
    'respect',
    'respectful',
    'respectfully',
    'respond',
    'responded',
    'response',
    'speak',
    'speaking',
    'tone',
}


def _normalize_grading_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text).lower()
    normalized = re.sub(r'([.!?;:,])(?=[^\s])', r'\1 ', normalized)
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def _canonical_content_token(token: str) -> str:
    if len(token) > 4:
        if token.endswith('ies'):
            token = token[:-3] + 'y'
        elif token.endswith('es') and not token.endswith('ses'):
            token = token[:-2]
        elif token.endswith('s') and not token.endswith('ss'):
            token = token[:-1]
    token = _CANONICAL_TOKEN_OVERRIDES.get(token, token)
    return token


def _content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw_token in _normalize_grading_text(text).split():
        token = _canonical_content_token(raw_token)
        if len(token) <= 1 or token in _GRADING_STOPWORDS:
            continue
        terms.add(token)
    return terms


def _obvious_reference_match_score(user_answer: str, reference_answer: str) -> int | None:
    normalized_user = _normalize_grading_text(user_answer)
    normalized_reference = _normalize_grading_text(reference_answer)
    if not normalized_user or not normalized_reference:
        return None
    if normalized_user == normalized_reference:
        return 10

    user_terms = _content_terms(user_answer)
    reference_terms = _content_terms(reference_answer)
    if not user_terms or not reference_terms:
        return None

    overlap = user_terms & reference_terms
    precision = len(overlap) / len(user_terms)
    recall = len(overlap) / len(reference_terms)
    sequence_ratio = SequenceMatcher(None, normalized_user, normalized_reference).ratio()

    if sequence_ratio >= 0.93 and precision >= 0.93 and recall >= 0.9:
        return 10
    if sequence_ratio >= 0.82 and precision >= 0.82 and recall >= 0.78:
        return 9
    return None


def _extract_quote_content(text: str) -> str:
    segments = [
        normalize_whitespace(segment.strip(' ,;:-'))
        for segment in re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', text)
        if normalize_whitespace(segment.strip(' ,;:-'))
    ]
    if not segments:
        return ''
    return normalize_whitespace(' '.join(segments))


def _is_dialogue_content_question(question: str, reference_answer: str) -> bool:
    lowered = question.lower().strip()
    has_quote_answer = bool(_extract_quote_content(reference_answer))
    if not has_quote_answer:
        return False
    if lowered.startswith('what did '):
        return True
    return lowered.startswith('how did ') and 'respond after hearing' in lowered


def _looks_like_manner_only_response(answer: str) -> bool:
    if '"' in answer or "'" in answer or '?' in answer:
        return False
    terms = _content_terms(answer)
    if not terms:
        return False
    substantive = terms - {'he', 'she', 'they', 'speaker'}
    return bool(substantive) and substantive <= _MANNER_ONLY_TERMS


def _context_sentences(context: str) -> list[str]:
    text_lines = [
        line.strip()
        for line in context.splitlines()
        if line.strip() and not line.lstrip().startswith('[Excerpt ')
    ]
    if not text_lines:
        return []
    sentences: list[str] = []
    for part in re.split(r'(?<=[.!?])\s+|;\s+', ' '.join(text_lines)):
        candidate = normalize_whitespace(part)
        if len(candidate) < 20:
            continue
        sentences.append(candidate)
    return list(dict.fromkeys(sentences))


def _best_evidence_sentences(question: str, reference_answer: str, context: str, *, limit: int = 3) -> list[str]:
    question_terms = _content_terms(question)
    reference_terms = _content_terms(_extract_quote_content(reference_answer) or reference_answer)
    ranked: list[tuple[float, str]] = []
    for sentence in _context_sentences(context):
        sentence_terms = _content_terms(sentence)
        if not sentence_terms:
            continue
        question_hits = len(question_terms & sentence_terms)
        reference_hits = len(reference_terms & sentence_terms)
        score = (question_hits * 1.35) + (reference_hits * 1.1)
        if score <= 0.0:
            continue
        ranked.append((score, sentence))
    ranked.sort(reverse=True)
    return [sentence for _score, sentence in ranked[:limit]]


def _evidence_support_metrics(user_answer: str, evidence_sentences: list[str]) -> tuple[float, float, float]:
    normalized_user = _normalize_grading_text(user_answer)
    if not normalized_user or not evidence_sentences:
        return 0.0, 0.0, 0.0

    user_terms = _content_terms(user_answer)
    user_quote = _extract_quote_content(user_answer)
    normalized_user_quote = _normalize_grading_text(user_quote) if user_quote else ''

    best_sequence = 0.0
    best_precision = 0.0
    best_quote_match = 0.0
    for sentence in evidence_sentences:
        normalized_sentence = _normalize_grading_text(sentence)
        if normalized_sentence:
            best_sequence = max(best_sequence, SequenceMatcher(None, normalized_user, normalized_sentence).ratio())
        sentence_terms = _content_terms(sentence)
        if user_terms and sentence_terms:
            overlap = user_terms & sentence_terms
            best_precision = max(best_precision, len(overlap) / max(len(user_terms), 1))
        evidence_quote = _extract_quote_content(sentence)
        normalized_evidence_quote = _normalize_grading_text(evidence_quote) if evidence_quote else ''
        if normalized_user_quote and normalized_evidence_quote:
            best_quote_match = max(
                best_quote_match,
                SequenceMatcher(None, normalized_user_quote, normalized_evidence_quote).ratio(),
            )
    return best_sequence, best_precision, best_quote_match


def _obvious_evidence_match_score(user_answer: str, evidence_sentences: list[str]) -> int | None:
    best_score: int | None = None
    for sentence in evidence_sentences:
        score = _obvious_reference_match_score(user_answer, sentence)
        if score is not None:
            best_score = max(best_score or 0, score)
        sentence_quote = _extract_quote_content(sentence)
        if sentence_quote:
            quote_score = _obvious_reference_match_score(user_answer, sentence_quote)
            if quote_score is not None:
                best_score = max(best_score or 0, quote_score)
    return best_score


def _embed_texts(embeddings, texts: list[str]) -> list[list[float]]:
    if embeddings is None or not texts:
        return []
    if hasattr(embeddings, 'embed_documents'):
        try:
            return embeddings.embed_documents(texts)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            pass
    return [embeddings.embed_query(text) for text in texts]


def _answer_support_metrics(user_answer: str, reference_answer: str, context: str) -> tuple[float, float, float, float]:
    normalized_user = _normalize_grading_text(user_answer)
    normalized_reference = _normalize_grading_text(reference_answer)
    if not normalized_user or not normalized_reference:
        return 0.0, 0.0, 0.0, 0.0

    user_terms = _content_terms(user_answer)
    reference_terms = _content_terms(reference_answer)
    context_terms = _content_terms(context)
    support_terms = context_terms | reference_terms
    overlap = user_terms & reference_terms
    precision = len(overlap) / max(len(user_terms), 1)
    recall = len(overlap) / max(len(reference_terms), 1)
    context_support = len(user_terms & support_terms) / max(len(user_terms), 1)
    sequence_ratio = SequenceMatcher(None, normalized_user, normalized_reference).ratio()
    return precision, recall, context_support, sequence_ratio


def _support_alignment_metrics(user_answer: str, reference_answer: str, context: str) -> tuple[float, float, float]:
    user_terms = _content_terms(user_answer)
    reference_terms = _content_terms(reference_answer)
    context_terms = _content_terms(context)
    if not user_terms:
        return 0.0, 0.0, 1.0

    reference_coverage = len(user_terms & reference_terms) / max(len(reference_terms), 1)
    extra_terms = user_terms - reference_terms
    if not extra_terms:
        return reference_coverage, 1.0, 0.0

    supported_extra = extra_terms & context_terms
    unsupported_extra = extra_terms - context_terms
    extra_support_ratio = len(supported_extra) / max(len(extra_terms), 1)
    unsupported_ratio = len(unsupported_extra) / max(len(user_terms), 1)
    return reference_coverage, extra_support_ratio, unsupported_ratio


def _heuristic_grade_floor(
    *,
    precision: float,
    recall: float,
    context_support: float,
    sequence_ratio: float,
    similarity: float,
) -> int:
    if recall >= 0.95 and precision >= 0.84 and context_support >= 0.72:
        return 10 if (sequence_ratio >= 0.9 or similarity >= 0.96) else 9
    if similarity >= 0.95 and recall >= 0.72 and precision >= 0.5 and context_support >= 0.5:
        return 9
    if recall >= 0.72 and precision >= 0.55 and context_support >= 0.68:
        return 8
    if similarity >= 0.9 and recall >= 0.58 and precision >= 0.42 and context_support >= 0.45:
        return 8
    if similarity >= 0.84 and recall >= 0.48 and precision >= 0.32 and context_support >= 0.35:
        return 7
    if recall >= 0.55 and context_support >= 0.4:
        return 6
    if recall >= 0.4 and context_support >= 0.3:
        return 5
    return 0


def _semantic_grade_floor(
    *,
    precision: float,
    recall: float,
    context_support: float,
    sequence_ratio: float,
    similarity: float,
    evidence_similarity: float,
) -> int:
    combined_similarity = max(similarity, evidence_similarity)
    if combined_similarity >= 0.96 and context_support >= 0.58:
        return 10 if (recall >= 0.5 or sequence_ratio >= 0.78) else 9
    if combined_similarity >= 0.96 and recall >= 0.45:
        return 8 if context_support >= 0.3 else 7
    if combined_similarity >= 0.92 and context_support >= 0.5:
        return 9
    if combined_similarity >= 0.88 and context_support >= 0.42:
        return 8 if (recall >= 0.24 or precision >= 0.2 or sequence_ratio >= 0.5) else 7
    if combined_similarity >= 0.82 and context_support >= 0.35:
        return 7
    if combined_similarity >= 0.76 and context_support >= 0.3:
        return 6
    if combined_similarity >= 0.68 and context_support >= 0.24:
        return 5
    return 0


def _dialogue_quote_recall(reference_answer: str, user_answer: str) -> float:
    quote_content = _extract_quote_content(reference_answer) or reference_answer
    quote_terms = _content_terms(quote_content)
    user_terms = _content_terms(user_answer)
    if not quote_terms or not user_terms:
        return 0.0
    return len(quote_terms & user_terms) / max(len(quote_terms), 1)


def _supported_equivalence_floor(
    *,
    question: str,
    reference_answer: str,
    user_answer: str,
    context_support: float,
    similarity: float,
    evidence_similarity: float,
    reference_evidence_similarity: float,
    reference_coverage: float,
    extra_support_ratio: float,
    unsupported_ratio: float,
    evidence_sequence: float,
    evidence_precision: float,
    evidence_quote_match: float,
) -> int:
    dialogue_quote_recall = 0.0
    if _is_dialogue_content_question(question, reference_answer):
        dialogue_quote_recall = _dialogue_quote_recall(reference_answer, user_answer)

    evidence_unavailable = evidence_similarity <= 0.05 and reference_evidence_similarity <= 0.05
    evidence_close_to_reference = evidence_unavailable or evidence_similarity >= max(0.62, reference_evidence_similarity - 0.1)
    strong_supported_expansion = (
        context_support >= 0.78
        and extra_support_ratio >= 0.82
        and unsupported_ratio <= 0.08
        and evidence_close_to_reference
    )
    if reference_coverage >= 0.92 and strong_supported_expansion:
        return 10
    if dialogue_quote_recall >= 0.9 and strong_supported_expansion:
        return 10
    if evidence_quote_match >= 0.96 and context_support >= 0.58 and unsupported_ratio <= 0.08:
        return 10
    if evidence_sequence >= 0.94 and evidence_precision >= 0.72 and context_support >= 0.62 and unsupported_ratio <= 0.08:
        return 10

    semantic_supported_expansion = (
        context_support >= 0.7
        and extra_support_ratio >= 0.75
        and unsupported_ratio <= 0.12
        and evidence_close_to_reference
    )
    if reference_coverage >= 0.72 and semantic_supported_expansion:
        return 10 if (similarity >= 0.5 or evidence_similarity >= 0.82) else 9
    if dialogue_quote_recall >= 0.68 and semantic_supported_expansion:
        return 10 if (evidence_similarity >= 0.7 or similarity >= 0.45) else 9
    if evidence_sequence >= 0.88 and evidence_precision >= 0.6 and context_support >= 0.54 and unsupported_ratio <= 0.12:
        return 9
    if reference_coverage >= 0.48 and semantic_supported_expansion:
        return 9
    return 0


def _dialogue_grade_floor(
    *,
    question: str,
    reference_answer: str,
    user_answer: str,
    context_support: float,
    recall: float,
    similarity: float,
    evidence_similarity: float,
) -> int | None:
    if not _is_dialogue_content_question(question, reference_answer):
        return None

    quote_terms = _content_terms(_extract_quote_content(reference_answer) or reference_answer)
    user_terms = _content_terms(user_answer)
    if not quote_terms or not user_terms:
        return None

    quote_recall = _dialogue_quote_recall(reference_answer, user_answer)
    combined_similarity = max(similarity, evidence_similarity)

    if _looks_like_manner_only_response(user_answer) and quote_recall < 0.25:
        return 2
    if quote_recall >= 0.92 and context_support >= 0.42:
        return 10 if combined_similarity >= 0.88 else 9
    if combined_similarity >= 0.96 and quote_recall >= 0.25:
        return 8
    if quote_recall >= 0.68 and combined_similarity >= 0.72:
        return 8
    if combined_similarity >= 0.9 and context_support >= 0.5:
        return 8
    if quote_recall >= 0.4 and context_support >= 0.32:
        return 6
    if combined_similarity >= 0.72 and (recall >= 0.18 or context_support >= 0.3):
        return 5
    if quote_recall > 0.0 or context_support >= 0.22:
        return 3
    return 0


def _fallback_score_from_similarity(similarity: float) -> int:
    if similarity >= 0.97:
        return 10
    if similarity >= 0.93:
        return 9
    if similarity >= 0.89:
        return 8
    if similarity >= 0.84:
        return 7
    if similarity >= 0.76:
        return 6
    if similarity >= 0.68:
        return 5
    if similarity >= 0.59:
        return 4
    if similarity >= 0.50:
        return 3
    if similarity >= 0.40:
        return 2
    if similarity > 0.0:
        return 1
    return 0


def _build_grading_context(
    store,
    config: AppConfig,
    question: GeneratedQuestion,
    source_chunks: list[Document] | None,
) -> str:
    if source_chunks and question.source_chunk_ids:
        selected = select_chunk_documents(source_chunks, question.source_chunk_ids, neighbor_window=2)
        if selected:
            return build_context(documents_to_retrieved_chunks(selected, relevance=1.0))
    if store is None:
        return ''

    retrieved = retrieve_chunks(
        store,
        question.question,
        config.retrieval.top_k,
        source_chunks=source_chunks,
    )
    return build_context(retrieved)


def grade_answer(
    llm,
    embeddings,
    store,
    config: AppConfig,
    question: GeneratedQuestion,
    user_answer: str,
    *,
    source_chunks: list[Document] | None = None,
) -> GradingResult:
    cleaned_user_answer = normalize_whitespace(user_answer)
    if not cleaned_user_answer:
        return GradingResult(score=0, feedback='No answer was provided.', model_answer=question.model_answer)
    obvious_match_score = _obvious_reference_match_score(cleaned_user_answer, question.model_answer)
    if obvious_match_score == 10:
        return GradingResult(
            score=10,
            feedback=_RUBRIC_FEEDBACK[10],
            model_answer=question.model_answer,
        )
    obvious_match_floor = obvious_match_score or 0

    context = _build_grading_context(store, config, question, source_chunks)
    evidence_sentences = _best_evidence_sentences(question.question, question.model_answer, context)
    obvious_evidence_match = _obvious_evidence_match_score(cleaned_user_answer, evidence_sentences)
    if obvious_evidence_match is not None and obvious_evidence_match >= 9:
        return GradingResult(
            score=10,
            feedback=_RUBRIC_FEEDBACK[10],
            model_answer=question.model_answer,
        )
    if embeddings is not None:
        embedded_vectors = _embed_texts(
            embeddings,
            [question.model_answer, cleaned_user_answer, *evidence_sentences],
        )
        reference_vector = embedded_vectors[0]
        answer_vector = embedded_vectors[1]
        evidence_vectors = embedded_vectors[2:]
        similarity = cosine_similarity(reference_vector, answer_vector)
        evidence_similarity = max(
            (cosine_similarity(answer_vector, vector) for vector in evidence_vectors),
            default=0.0,
        )
        reference_evidence_similarity = max(
            (cosine_similarity(reference_vector, vector) for vector in evidence_vectors),
            default=0.0,
        )
    else:
        similarity = 0.0
        evidence_similarity = 0.0
        reference_evidence_similarity = 0.0
    evidence_sequence, evidence_precision, evidence_quote_match = _evidence_support_metrics(
        cleaned_user_answer,
        evidence_sentences,
    )
    precision, recall, context_support, sequence_ratio = _answer_support_metrics(
        cleaned_user_answer,
        question.model_answer,
        context,
    )
    reference_coverage, extra_support_ratio, unsupported_ratio = _support_alignment_metrics(
        cleaned_user_answer,
        question.model_answer,
        context,
    )
    heuristic_floor = _heuristic_grade_floor(
        precision=precision,
        recall=recall,
        context_support=context_support,
        sequence_ratio=sequence_ratio,
        similarity=similarity,
    )
    semantic_floor = _semantic_grade_floor(
        precision=precision,
        recall=recall,
        context_support=context_support,
        sequence_ratio=sequence_ratio,
        similarity=similarity,
        evidence_similarity=evidence_similarity,
    )
    supported_equivalence_floor = _supported_equivalence_floor(
        question=question.question,
        reference_answer=question.model_answer,
        user_answer=cleaned_user_answer,
        context_support=context_support,
        similarity=similarity,
        evidence_similarity=evidence_similarity,
        reference_evidence_similarity=reference_evidence_similarity,
        reference_coverage=reference_coverage,
        extra_support_ratio=extra_support_ratio,
        unsupported_ratio=unsupported_ratio,
        evidence_sequence=evidence_sequence,
        evidence_precision=evidence_precision,
        evidence_quote_match=evidence_quote_match,
    )
    dialogue_floor = _dialogue_grade_floor(
        question=question.question,
        reference_answer=question.model_answer,
        user_answer=cleaned_user_answer,
        context_support=context_support,
        recall=recall,
        similarity=similarity,
        evidence_similarity=evidence_similarity,
    )
    heuristic_floor = max(
        obvious_match_floor,
        obvious_evidence_match or 0,
        heuristic_floor,
        semantic_floor,
        supported_equivalence_floor,
        dialogue_floor or 0,
    )
    if heuristic_floor >= 8 or heuristic_floor <= 2:
        return GradingResult(
            score=heuristic_floor,
            feedback=_RUBRIC_FEEDBACK[heuristic_floor],
            model_answer=question.model_answer,
        )
    if heuristic_floor >= 9:
        return GradingResult(
            score=heuristic_floor,
            feedback=_RUBRIC_FEEDBACK[heuristic_floor],
            model_answer=question.model_answer,
        )
    if llm is None:
        return GradingResult(
            score=max(heuristic_floor, 0),
            feedback=_RUBRIC_FEEDBACK[max(heuristic_floor, 0)],
            model_answer=question.model_answer,
        )

    system_prompt = (
        'Grade the student answer using only the supplied context and reference answer. '
        'Accept correct paraphrases and supported alternative wording. '
        'Do not require the student to match the reference wording. '
        'Do not penalize extra correct information unless it replaces, distorts, or contradicts the asked fact. '
        'If the student adds correct supported explanation beyond the reference answer, deduct at most one or two points. '
        'For dialogue questions, grade the actual meaning of what was said, not whether the student copied the exact quote scaffold. '
        'If the student answer is broadly correct but incomplete, prefer the middle bands instead of the failure bands. '
        'Reserve 0, 1, and 2 for blank, irrelevant, contradicted, or almost entirely unsupported answers. '
        'Use this exact rubric: '
        '10 = completely correct and complete; '
        '9 = correct and nearly complete, tiny wording/detail gap; '
        '8 = clearly correct on the core answer, small omission/imprecision; '
        '7 = mostly correct, one notable detail missing or slightly wrong; '
        '6 = more right than wrong, main idea present but major detail missing or underspecified; '
        '5 = partially correct, relevant understanding is present but important parts are missing or unclear; '
        '4 = limited partial answer, on topic and somewhat supported but missing the main asked detail; '
        '3 = only slight overlap with the source-backed answer; '
        '2 = almost entirely incorrect though loosely on topic; '
        '1 = incorrect and unsupported; '
        '0 = blank, irrelevant, or directly contradicted by the text. '
        'Give 9 or 10 when the answer is fully or nearly fully correct, even if it uses different wording. '
        'Return valid JSON only with this schema: '
        '{"score":0,"feedback":"...","model_answer":"..."}. '
        'feedback must be one short sentence that states what was correct and what was missing or wrong.'
    )
    user_prompt = (
        f'Question: {question.question}\n\n'
        f'Context:\n{context}\n\n'
        f'Reference answer: {question.model_answer}\n\n'
        f'Student answer: {cleaned_user_answer}\n\n'
        'Grade against the context first and use the reference answer as a concise canonical answer.'
    )
    response = llm.invoke([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ])
    raw_text = coerce_content(response)

    try:
        payload = extract_json_object(raw_text)
        score = int(payload.get('score', 0))
        score = max(0, min(10, score))
        feedback = normalize_whitespace(str(payload.get('feedback', '')))
        score = max(score, heuristic_floor)
        if feedback:
            if score == heuristic_floor and score > int(payload.get('score', 0)):
                feedback = _RUBRIC_FEEDBACK[score]
            return GradingResult(score=score, feedback=feedback, model_answer=question.model_answer)
    except (TypeError, ValueError):
        pass

    reference_terms = set(normalize_question_text(question.model_answer).split())
    answer_terms = set(normalize_question_text(cleaned_user_answer).split())
    term_recall = len(reference_terms & answer_terms) / max(len(reference_terms), 1)
    context_terms: set[str] = set()
    for line in context.splitlines():
        context_terms.update(normalize_question_text(line).split())
    context_recall = len(context_terms & answer_terms) / max(len(answer_terms), 1)
    heuristic = max(similarity, evidence_similarity, (similarity * 0.55) + (context_recall * 0.25) + (term_recall * 0.2))
    score = max(_fallback_score_from_similarity(heuristic), heuristic_floor)
    return GradingResult(
        score=score,
        feedback=_RUBRIC_FEEDBACK[score],
        model_answer=question.model_answer,
    )
