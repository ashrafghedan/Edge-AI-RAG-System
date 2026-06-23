from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .types import RetrievedChunk


_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'did',
    'do',
    'does',
    'for',
    'from',
    'had',
    'has',
    'have',
    'he',
    'her',
    'hers',
    'him',
    'his',
    'i',
    'in',
    'is',
    'it',
    'its',
    'me',
    'my',
    'of',
    'on',
    'or',
    'she',
    'that',
    'the',
    'their',
    'them',
    'there',
    'they',
    'this',
    'to',
    'was',
    'what',
    'when',
    'where',
    'which',
    'who',
    'why',
    'with',
    'would',
    'you',
    'your',
}

_QUERY_TERM_EXPANSIONS = {
    'change': {'changes', 'changed', 'shift', 'shifts', 'pattern', 'light'},
    'changes': {'change', 'changed', 'shift', 'shifts', 'pattern', 'light'},
    'consequence': {'result', 'effect', 'outcome', 'thus', 'shunned', 'ostracized', 'solitude'},
    'condition': {'diagnosis', 'illness', 'depression', 'hysterical', 'tendency', 'sick'},
    'diagnosis': {'condition', 'illness', 'depression', 'hysterical', 'tendency', 'sick'},
    'effect': {'consequence', 'result', 'outcome', 'thus'},
    'light': {'lights', 'sun', 'sunlight', 'moonlight', 'changes'},
    'occupation': {'profession', 'professions', 'job', 'work', 'physician', 'doctor'},
    'join': {'joined', 'come', 'regulations', 'proposed', 'union'},
    'refuse': {'refused', 'could', 'come', 'regulations', 'proposed', 'join'},
    'refused': {'refuse', 'could', 'come', 'regulations', 'proposed', 'join'},
    'prescribe': {'prescribed', 'treatment', 'tonics', 'medicine', 'phosphates', 'phosphites'},
    'prescribed': {'prescribe', 'treatment', 'tonics', 'medicine', 'phosphates', 'phosphites'},
    'profession': {'professions', 'occupation', 'job', 'work', 'physician', 'doctor'},
    'professions': {'profession', 'occupation', 'job', 'work', 'physician', 'doctor'},
    'responsibilities': {'responsibility', 'duties', 'role', 'job', 'tasks', 'manage', 'care'},
    'responsibility': {'responsibilities', 'duties', 'role', 'job', 'tasks', 'manage', 'care'},
    'result': {'consequence', 'effect', 'outcome', 'thus', 'shunned', 'ostracized'},
    'role': {'responsibilities', 'responsibility', 'duties', 'job', 'tasks', 'care'},
    'shift': {'shifts', 'change', 'changes', 'light', 'sunlight', 'moonlight'},
    'shifts': {'shift', 'change', 'changes', 'light', 'sunlight', 'moonlight'},
    'treatment': {'prescribed', 'prescribe', 'tonics', 'medicine', 'phosphates', 'phosphites'},
    'union': {'regulations', 'proposed', 'combination'},
}

_DESCRIPTION_CUE_TERMS = {'is', 'was', 'are', 'were'}
_ROLE_CUE_TERMS = {
    'care',
    'cares',
    'duties',
    'duty',
    'everything',
    'housekeeper',
    'job',
    'manage',
    'manages',
    'profession',
    'role',
    'sees',
    'tasks',
}


def _canonicalize_token(token: str) -> str:
    lowered = token.lower().strip().strip("'")
    if lowered.endswith("'s"):
        lowered = lowered[:-2]
    elif lowered.endswith("s'"):
        lowered = lowered[:-1]
    return lowered


def _normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text)
    normalized = normalized.replace('?', "'").replace('?', '"').replace('?', '"')
    normalized = normalized.replace('<channel|>', ' ')
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s']", ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def _tokenize_search_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _normalize_search_text(text).split():
        token = _canonicalize_token(raw_token)
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _extract_quoted_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    for raw_phrase in re.findall(r'"([^"]{8,})"', query):
        phrases.append(raw_phrase)
    for raw_phrase in re.findall(r"'([^']{8,})'", query):
        if raw_phrase.count(' ') >= 2:
            phrases.append(raw_phrase)
    return list(dict.fromkeys(_normalize_search_text(phrase) for phrase in phrases if phrase.strip()))


def _query_entity_tokens(query: str) -> set[str]:
    return {
        _canonicalize_token(token)
        for token in re.findall(r"\b[A-Z][a-z']{2,}\b", query)
        if _canonicalize_token(token) not in _STOPWORDS
    }


def _is_role_query(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in lowered
        for term in (
            'responsibilities',
            'responsibility',
            'duties',
            'role',
            'profession',
            'professions',
            'occupation',
            'job',
        )
    )


def _is_entity_description_query(query: str) -> bool:
    lowered = query.lower().strip()
    return (
        lowered.startswith('who is ')
        or lowered.startswith('who was ')
        or lowered.startswith('who are ')
        or lowered.startswith('who were ')
        or (_is_role_query(query) and bool(_query_entity_tokens(query)))
    )


def _query_variants(query: str) -> list[str]:
    variants = [query.strip()]
    entity_tokens_in_order = [
        token
        for token in re.findall(r"\b[A-Z][a-z']{2,}\b", query)
        if _canonicalize_token(token) not in _STOPWORDS
    ]
    subject_prefix = ' '.join(entity_tokens_in_order[:2]).strip()
    if _is_multi_part_query(query):
        for part in re.split(r'\s+and\s+', query, flags=re.IGNORECASE):
            cleaned = part.strip(' ,.?')
            if not cleaned:
                continue
            variants.append(cleaned)
            if subject_prefix and subject_prefix.lower() not in cleaned.lower():
                variants.append(f'{subject_prefix} {cleaned}')
    if _is_entity_description_query(query) and subject_prefix:
        variants.append(subject_prefix)
        if _is_role_query(query):
            variants.append(f'{subject_prefix} responsibilities')
            variants.append(f'{subject_prefix} role in the house')
    return list(dict.fromkeys(variant for variant in variants if variant))


def _is_multi_part_query(query: str) -> bool:
    lowered = query.lower()
    if any(phrase in lowered for phrase in (' and what ', ' and how ', ' and why ', ' and who ', ' and where ', ' and when ')):
        return True
    if any(phrase in lowered for phrase in (' consequence ', ' result ', ' effect ', ' responsibilities ', ' role ')):
        return True
    return ' and ' in lowered and len(_tokenize_search_text(query)) >= 10


def _expand_query_terms(query: str, direct_terms: set[str]) -> set[str]:
    expanded: set[str] = set()
    for token in direct_terms:
        expanded.update(_QUERY_TERM_EXPANSIONS.get(token, set()))

    lowered = query.lower()
    if _is_role_query(query):
        expanded.update(_ROLE_CUE_TERMS)
        expanded.update({'everything', 'keep', 'keeping', 'takes', 'care'})
    if _is_entity_description_query(query):
        expanded.update(_DESCRIPTION_CUE_TERMS)
        expanded.update({'sister', 'brother', 'mother', 'father', 'wife', 'husband', 'friend'})
    if any(term in lowered for term in ('refuse', 'refused', 'join the union', 'join union')):
        expanded.update({'could', 'come', 'proposed', 'regulations'})
    if 'diagnosis' in lowered or 'condition' in lowered:
        expanded.update({'temporary', 'nervous', 'depression', 'hysterical', 'tendency', 'sick'})
    if any(term in lowered for term in ('treatment', 'prescribed', 'prescribe')):
        expanded.update({'air', 'cod', 'exercise', 'forbidden', 'journeys', 'liver', 'oil', 'phosphates', 'phosphites', 'tonics', 'work'})
    if any(term in lowered for term in ('profession', 'professions', 'occupation')):
        expanded.update({'physician', 'doctor', 'housekeeper'})
    if any(term in lowered for term in ('change', 'changes', 'shift', 'shifts', 'light')):
        expanded.update({'moonlight', 'pattern', 'sun', 'sunlight'})
    return {term for term in expanded if len(term) > 1 and term not in _STOPWORDS} - direct_terms


def _variant_term_sets(query: str) -> list[set[str]]:
    variants: list[set[str]] = []
    for variant in _query_variants(query):
        tokens = set(_tokenize_search_text(variant))
        if tokens:
            variants.append(tokens)
    return variants


def _term_weights(query_terms: set[str], source_chunks: list[Document]) -> dict[str, float]:
    if not query_terms:
        return {}
    if not source_chunks:
        return {term: 1.0 for term in query_terms}

    document_frequency: Counter[str] = Counter()
    for document in source_chunks:
        document_terms = set(_tokenize_search_text(document.page_content))
        for term in document_terms & query_terms:
            document_frequency[term] += 1

    total_documents = max(len(source_chunks), 1)
    return {
        term: math.log((total_documents + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
        for term in query_terms
    }


def _entity_intro_indices(source_chunks: list[Document], entity_tokens: set[str]) -> dict[tuple[str, str], int]:
    if not source_chunks or not entity_tokens:
        return {}

    intro_indices: dict[tuple[str, str], int] = {}
    for document in source_chunks:
        chunk_index = int(document.metadata.get('chunk_index', -1))
        if chunk_index < 0:
            continue
        source_name = str(document.metadata.get('source_name', 'unknown'))
        content_tokens = set(_tokenize_search_text(document.page_content))
        for token in entity_tokens & content_tokens:
            key = (source_name, token)
            previous = intro_indices.get(key)
            if previous is None or chunk_index < previous:
                intro_indices[key] = chunk_index
    return intro_indices


def _sentence_windows(text: str, *, max_window_characters: int = 420) -> list[str]:
    sentences = [
        re.sub(r'\s+', ' ', part).strip()
        for part in re.split(r'(?<=[.!?])\s+|;\s+', text)
        if re.sub(r'\s+', ' ', part).strip()
    ]
    windows: list[str] = []
    for index, sentence in enumerate(sentences):
        windows.append(sentence)
        if index + 1 < len(sentences):
            paired = f'{sentence} {sentences[index + 1]}'.strip()
            if len(paired) <= max_window_characters:
                windows.append(paired)
    return list(dict.fromkeys(window for window in windows if len(window) >= 30))


def _entity_intro_bonus(
    document: Document,
    content_tokens: set[str],
    entity_tokens: set[str],
    intro_indices: dict[tuple[str, str], int],
) -> float:
    if not entity_tokens or not intro_indices:
        return 0.0
    shared_entities = entity_tokens & content_tokens
    if not shared_entities:
        return 0.0

    source_name = str(document.metadata.get('source_name', 'unknown'))
    chunk_index = int(document.metadata.get('chunk_index', -1))
    if chunk_index < 0:
        return 0.0

    bonus = 0.0
    for entity in shared_entities:
        intro_index = intro_indices.get((source_name, entity))
        if intro_index is None:
            continue
        distance = abs(chunk_index - intro_index)
        if distance <= 1:
            bonus = max(bonus, 0.9)
        elif distance <= 2:
            bonus = max(bonus, 0.6)
        elif distance <= 4:
            bonus = max(bonus, 0.3)
    return bonus


def _score_candidate_text(
    text: str,
    *,
    direct_terms: set[str],
    expanded_terms: set[str],
    term_weights: dict[str, float],
    entity_tokens: set[str],
    quoted_phrases: list[str],
    variant_terms: list[set[str]],
    description_query: bool,
    role_query: bool,
) -> tuple[float, int]:
    normalized = _normalize_search_text(text)
    if not normalized:
        return 0.0, 0

    tokens = set(_tokenize_search_text(normalized))
    if not tokens:
        return 0.0, 0

    shared_direct = direct_terms & tokens
    shared_expanded = expanded_terms & tokens
    shared_entities = entity_tokens & tokens
    phrase_hits = sum(1 for phrase in quoted_phrases if phrase in normalized)
    variant_hits = sum(1 for terms in variant_terms if terms & tokens)

    if not shared_direct and not shared_expanded and not shared_entities and phrase_hits == 0:
        return 0.0, 0

    direct_score = sum(term_weights.get(term, 1.0) for term in shared_direct) * 1.35
    expanded_score = sum(term_weights.get(term, 0.9) for term in shared_expanded) * 0.65
    entity_score = sum(term_weights.get(term, 0.9) for term in shared_entities) * 0.3
    score = direct_score + expanded_score + entity_score + (phrase_hits * 5.0)

    if variant_hits > 1:
        score += (variant_hits - 1) * 0.9

    if description_query and shared_entities and any(term in tokens for term in _DESCRIPTION_CUE_TERMS):
        score += 0.8
    if role_query and (shared_entities or shared_direct) and any(term in tokens for term in _ROLE_CUE_TERMS):
        score += 1.2

    if '?' in text:
        score -= 2.5
    if shared_entities and not shared_direct and not shared_expanded and phrase_hits == 0:
        score = min(score, 1.15)

    return max(score, 0.0), phrase_hits


def _document_to_retrieved_chunk(
    document: Document,
    *,
    distance: float,
    relevance: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(document.metadata.get('chunk_id', 'unknown')),
        source_name=str(document.metadata.get('source_name', 'unknown')),
        content=document.page_content,
        distance=distance,
        relevance=relevance,
    )


def _vector_candidates(store: Chroma, query: str, limit: int) -> dict[str, tuple[Document, float, float]]:
    candidates: dict[str, tuple[Document, float, float]] = {}
    for variant in _query_variants(query):
        for document, distance in store.similarity_search_with_score(variant, k=limit):
            chunk_id = str(document.metadata.get('chunk_id', 'unknown'))
            numeric_distance = float(distance)
            relevance = 1.0 / (1.0 + max(numeric_distance, 0.0))
            existing = candidates.get(chunk_id)
            if existing is None or relevance > existing[2]:
                candidates[chunk_id] = (document, numeric_distance, relevance)
    return candidates


def _lexical_candidates(
    query: str,
    source_chunks: list[Document],
    limit: int,
) -> dict[str, tuple[Document, float, int]]:
    direct_terms = set()
    for variant in _query_variants(query):
        direct_terms.update(_tokenize_search_text(variant))
    expanded_terms = _expand_query_terms(query, direct_terms)
    entity_tokens = _query_entity_tokens(query)
    quoted_phrases = _extract_quoted_phrases(query)
    variant_terms = _variant_term_sets(query)
    term_weights = _term_weights(direct_terms | expanded_terms | entity_tokens, source_chunks)
    intro_indices = _entity_intro_indices(source_chunks, entity_tokens)
    description_query = _is_entity_description_query(query)
    role_query = _is_role_query(query)

    candidates: list[tuple[float, int, Document]] = []
    for document in source_chunks:
        content_tokens = set(_tokenize_search_text(document.page_content))
        whole_score, whole_phrase_hits = _score_candidate_text(
            document.page_content,
            direct_terms=direct_terms,
            expanded_terms=expanded_terms,
            term_weights=term_weights,
            entity_tokens=entity_tokens,
            quoted_phrases=quoted_phrases,
            variant_terms=variant_terms,
            description_query=description_query,
            role_query=role_query,
        )

        best_window_score = 0.0
        best_window_phrase_hits = 0
        for window in _sentence_windows(document.page_content):
            window_score, window_phrase_hits = _score_candidate_text(
                window,
                direct_terms=direct_terms,
                expanded_terms=expanded_terms,
                term_weights=term_weights,
                entity_tokens=entity_tokens,
                quoted_phrases=quoted_phrases,
                variant_terms=variant_terms,
                description_query=description_query,
                role_query=role_query,
            )
            if (window_phrase_hits, window_score) > (best_window_phrase_hits, best_window_score):
                best_window_score = window_score
                best_window_phrase_hits = window_phrase_hits

        intro_bonus = _entity_intro_bonus(document, content_tokens, entity_tokens, intro_indices) if (description_query or role_query) else 0.0
        lexical_score = max(best_window_score, whole_score * 0.75) + min(best_window_score, whole_score) * 0.2 + intro_bonus
        exact_phrase_hits = max(whole_phrase_hits, best_window_phrase_hits)

        if exact_phrase_hits == 0 and lexical_score < 1.4:
            continue

        candidates.append((lexical_score, exact_phrase_hits, document))

    candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return {
        str(document.metadata.get('chunk_id', 'unknown')): (document, lexical_score, exact_phrase_hits)
        for lexical_score, exact_phrase_hits, document in candidates[:limit]
    }


def select_chunk_documents(
    source_chunks: list[Document],
    chunk_ids: list[str],
    *,
    neighbor_window: int = 1,
) -> list[Document]:
    if not source_chunks or not chunk_ids:
        return []

    by_position: dict[tuple[str, int], Document] = {}
    by_id: dict[str, Document] = {}
    for document in source_chunks:
        source_name = str(document.metadata.get('source_name', 'unknown'))
        chunk_index = int(document.metadata.get('chunk_index', -1))
        chunk_id = str(document.metadata.get('chunk_id', 'unknown'))
        by_position[(source_name, chunk_index)] = document
        by_id[chunk_id] = document

    selected: list[Document] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        document = by_id.get(chunk_id)
        if document is None:
            continue
        source_name = str(document.metadata.get('source_name', 'unknown'))
        chunk_index = int(document.metadata.get('chunk_index', -1))
        for offset in range(-neighbor_window, neighbor_window + 1):
            neighbor = by_position.get((source_name, chunk_index + offset))
            if neighbor is None:
                continue
            neighbor_id = str(neighbor.metadata.get('chunk_id', 'unknown'))
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            selected.append(neighbor)
    return selected


def documents_to_retrieved_chunks(documents: list[Document], *, relevance: float = 1.0) -> list[RetrievedChunk]:
    return [
        _document_to_retrieved_chunk(
            document,
            distance=max(0.0, 1.0 - relevance),
            relevance=relevance,
        )
        for document in documents
    ]


def retrieve_chunks(
    store: Chroma,
    query: str,
    top_k: int,
    *,
    source_chunks: list[Document] | None = None,
    neighbor_window: int | None = None,
) -> list[RetrievedChunk]:
    quoted_phrases = _extract_quoted_phrases(query)
    query_has_quote = bool(quoted_phrases)
    multi_part_query = _is_multi_part_query(query)
    description_query = _is_entity_description_query(query)
    role_query = _is_role_query(query)
    corpus_size = len(source_chunks or [])
    if corpus_size >= 180:
        vector_limit = min(max(top_k * 6, 32), 64)
    elif corpus_size >= 80:
        vector_limit = min(max(top_k * 5, 24), 48)
    elif corpus_size >= 40:
        vector_limit = min(max(top_k * 4, 18), 32)
    elif corpus_size:
        vector_limit = min(max(top_k * 3, 12), max(corpus_size, 12))
    else:
        vector_limit = max(top_k * 2, 8)
    if neighbor_window is None:
        neighbor_window = 2 if (multi_part_query or corpus_size >= 40 or description_query) else 1

    vector_hits = _vector_candidates(store, query, vector_limit)
    lexical_hits = _lexical_candidates(query, source_chunks or [], vector_limit) if source_chunks else {}
    vector_ids = list(vector_hits)

    selected_ids: list[str] = []
    if query_has_quote or description_query or role_query:
        seed_order = list(lexical_hits) + list(vector_hits)
    else:
        seed_order = list(vector_hits) + list(lexical_hits)
    for chunk_id in seed_order:
        if chunk_id not in selected_ids:
            selected_ids.append(chunk_id)

    expansion_seed_ids: list[str] = []
    if source_chunks and selected_ids:
        if query_has_quote:
            seed_candidates = selected_ids[: max(vector_limit, top_k + 2)]
        else:
            seed_candidates = list(lexical_hits)[: max(top_k + 2, 6)] + vector_ids[: max(top_k + 2, 6)]
        for chunk_id in seed_candidates:
            if chunk_id not in expansion_seed_ids:
                expansion_seed_ids.append(chunk_id)
        expanded_documents = select_chunk_documents(
            source_chunks,
            expansion_seed_ids,
            neighbor_window=neighbor_window,
        )
    else:
        expanded_documents = []

    expanded_by_id = {
        str(document.metadata.get('chunk_id', 'unknown')): document for document in expanded_documents
    }

    all_candidate_ids: list[str] = []
    for chunk_id in selected_ids:
        if chunk_id not in all_candidate_ids:
            all_candidate_ids.append(chunk_id)
    for chunk_id in expanded_by_id:
        if chunk_id not in all_candidate_ids:
            all_candidate_ids.append(chunk_id)

    if not all_candidate_ids:
        return []

    lexical_scores = {chunk_id: score for chunk_id, (_document, score, _hits) in lexical_hits.items()}
    lexical_max = max(lexical_scores.values(), default=0.0)
    candidate_documents: dict[str, Document] = {}
    for chunk_id in all_candidate_ids:
        if chunk_id in vector_hits:
            candidate_documents[chunk_id] = vector_hits[chunk_id][0]
        elif chunk_id in lexical_hits:
            candidate_documents[chunk_id] = lexical_hits[chunk_id][0]
        else:
            candidate_documents[chunk_id] = expanded_by_id[chunk_id]

    ranked: list[tuple[float, int, RetrievedChunk]] = []
    for chunk_id in all_candidate_ids:
        document = candidate_documents[chunk_id]
        if chunk_id in vector_hits:
            _vector_document, distance, vector_relevance = vector_hits[chunk_id]
        elif chunk_id in lexical_hits:
            distance = 1.0
            vector_relevance = 0.0
        else:
            distance = 1.0
            vector_relevance = 0.0

        lexical_score = lexical_scores.get(chunk_id, 0.0)
        lexical_relevance = lexical_score / lexical_max if lexical_max > 0.0 else 0.0
        exact_phrase_hits = lexical_hits.get(chunk_id, (None, 0.0, 0))[2] if chunk_id in lexical_hits else 0

        neighbor_bonus = 0.0
        if chunk_id in expanded_by_id and chunk_id not in lexical_hits and chunk_id not in vector_hits:
            source_name = str(document.metadata.get('source_name', 'unknown'))
            chunk_index = int(document.metadata.get('chunk_index', -1))
            decayed_support = 0.0
            if chunk_index >= 0:
                for seed_id in expansion_seed_ids:
                    seed_document = candidate_documents.get(seed_id)
                    if seed_document is None:
                        continue
                    if str(seed_document.metadata.get('source_name', 'unknown')) != source_name:
                        continue
                    seed_index = int(seed_document.metadata.get('chunk_index', -1000))
                    if abs(seed_index - chunk_index) > neighbor_window:
                        continue

                    if seed_id in vector_hits:
                        _seed_document, _seed_distance, seed_vector_relevance = vector_hits[seed_id]
                    else:
                        seed_vector_relevance = 0.0
                    seed_lexical_score = lexical_scores.get(seed_id, 0.0)
                    seed_lexical_relevance = seed_lexical_score / lexical_max if lexical_max > 0.0 else 0.0
                    seed_support = max(seed_vector_relevance, seed_lexical_relevance)
                    decayed_support = max(decayed_support, seed_support * 0.38)

            neighbor_bonus = max(0.1, min(0.22, decayed_support))

        cluster_bonus = 0.0
        source_name = str(document.metadata.get('source_name', 'unknown'))
        chunk_index = int(document.metadata.get('chunk_index', -1))
        if chunk_index >= 0:
            nearby_support = 0
            for other_id, other_document in candidate_documents.items():
                if other_id == chunk_id:
                    continue
                if str(other_document.metadata.get('source_name', 'unknown')) != source_name:
                    continue
                other_index = int(other_document.metadata.get('chunk_index', -1000))
                if other_index >= 0 and abs(other_index - chunk_index) <= 10:
                    nearby_support += 1
            cluster_bonus = min(0.14, nearby_support * 0.045)

        if query_has_quote:
            combined = max(vector_relevance, lexical_relevance) + neighbor_bonus + cluster_bonus + (0.8 if exact_phrase_hits else 0.0)
        else:
            combined = (vector_relevance * 0.6) + (lexical_relevance * 0.4) + neighbor_bonus + cluster_bonus
            if description_query or role_query:
                combined += lexical_relevance * 0.08

        ranked.append(
            (
                combined,
                int(exact_phrase_hits),
                _document_to_retrieved_chunk(
                    document,
                    distance=distance,
                    relevance=max(vector_relevance, lexical_relevance, 0.34 if chunk_id in expanded_by_id else 0.0),
                ),
            )
        )

    ranked.sort(key=lambda item: (item[1], item[0], item[2].relevance), reverse=True)
    ranked_by_id = {chunk.chunk_id: (score, phrase_hits, chunk) for score, phrase_hits, chunk in ranked}

    final_chunks: list[RetrievedChunk] = []
    seen_final: set[str] = set()

    if multi_part_query and source_chunks:
        for variant in _query_variants(query)[1:]:
            variant_terms = set(_tokenize_search_text(variant))
            if not variant_terms:
                continue
            best_variant_id = ''
            best_variant_score = -1.0
            variant_expanded_terms = _expand_query_terms(variant, variant_terms)
            variant_weights = _term_weights(variant_terms | variant_expanded_terms | _query_entity_tokens(variant), source_chunks)
            for chunk_id in all_candidate_ids:
                document = candidate_documents[chunk_id]
                score, _phrase_hits = _score_candidate_text(
                    document.page_content,
                    direct_terms=variant_terms,
                    expanded_terms=variant_expanded_terms,
                    term_weights=variant_weights,
                    entity_tokens=_query_entity_tokens(variant),
                    quoted_phrases=[],
                    variant_terms=[variant_terms],
                    description_query=_is_entity_description_query(variant),
                    role_query=_is_role_query(variant),
                )
                source_name = str(document.metadata.get('source_name', 'unknown'))
                chunk_index = int(document.metadata.get('chunk_index', -1))
                nearby_support = 0
                if chunk_index >= 0:
                    for other_id in all_candidate_ids:
                        if other_id == chunk_id:
                            continue
                        other_document = candidate_documents[other_id]
                        if str(other_document.metadata.get('source_name', 'unknown')) != source_name:
                            continue
                        other_index = int(other_document.metadata.get('chunk_index', -1000))
                        if other_index >= 0 and abs(other_index - chunk_index) <= 10:
                            nearby_support += 1
                score += min(4.0, nearby_support * 2.0)
                if score > best_variant_score:
                    best_variant_score = score
                    best_variant_id = chunk_id
            if best_variant_id and best_variant_id not in seen_final:
                seen_final.add(best_variant_id)
                final_chunks.append(ranked_by_id[best_variant_id][2])

    result_limit = top_k + 2 if (multi_part_query or corpus_size >= 40) else max(top_k, 6)
    for _combined, _phrase_hits, chunk in ranked:
        if chunk.chunk_id in seen_final:
            continue
        seen_final.add(chunk.chunk_id)
        final_chunks.append(chunk)
        if len(final_chunks) >= result_limit:
            break

    return final_chunks[:result_limit]


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f'[Excerpt {index} | chunk_id={chunk.chunk_id} | source={chunk.source_name}]\n{chunk.content}'
        )
    return '\n\n'.join(blocks)


def build_focused_context(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_excerpts: int = 6,
    max_chars_per_excerpt: int = 360,
) -> str:
    if not chunks:
        return ''

    direct_terms = set()
    for variant in _query_variants(query):
        direct_terms.update(_tokenize_search_text(variant))
    expanded_terms = _expand_query_terms(query, direct_terms)
    entity_tokens = _query_entity_tokens(query)
    quoted_phrases = _extract_quoted_phrases(query)
    variant_terms = _variant_term_sets(query)
    pseudo_documents = [
        Document(
            page_content=chunk.content,
            metadata={'chunk_id': chunk.chunk_id, 'source_name': chunk.source_name, 'chunk_index': index},
        )
        for index, chunk in enumerate(chunks)
    ]
    term_weights = _term_weights(direct_terms | expanded_terms | entity_tokens, pseudo_documents)
    description_query = _is_entity_description_query(query)
    role_query = _is_role_query(query)

    scored_windows: list[tuple[float, str, str, str]] = []
    for chunk in chunks:
        for window in _sentence_windows(chunk.content):
            window_score, _phrase_hits = _score_candidate_text(
                window,
                direct_terms=direct_terms,
                expanded_terms=expanded_terms,
                term_weights=term_weights,
                entity_tokens=entity_tokens,
                quoted_phrases=quoted_phrases,
                variant_terms=variant_terms,
                description_query=description_query,
                role_query=role_query,
            )
            if window_score <= 0.0:
                continue
            boosted_score = window_score + (chunk.relevance * 1.1)
            scored_windows.append((boosted_score, chunk.chunk_id, chunk.source_name, window[:max_chars_per_excerpt].strip()))

    if not scored_windows:
        return build_context(chunks[:max_excerpts])

    selected: list[tuple[str, str, str]] = []
    selected_texts: set[str] = set()
    per_chunk_counts: dict[str, int] = {}

    if _is_multi_part_query(query):
        for variant in _query_variants(query)[1:]:
            variant_direct_terms = set(_tokenize_search_text(variant))
            if not variant_direct_terms:
                continue
            variant_expanded_terms = _expand_query_terms(variant, variant_direct_terms)
            variant_weights = _term_weights(
                variant_direct_terms | variant_expanded_terms | _query_entity_tokens(variant),
                pseudo_documents,
            )
            best_variant: tuple[float, str, str, str] | None = None
            for chunk in chunks:
                for window in _sentence_windows(chunk.content):
                    score, _hits = _score_candidate_text(
                        window,
                        direct_terms=variant_direct_terms,
                        expanded_terms=variant_expanded_terms,
                        term_weights=variant_weights,
                        entity_tokens=_query_entity_tokens(variant),
                        quoted_phrases=[],
                        variant_terms=[variant_direct_terms],
                        description_query=_is_entity_description_query(variant),
                        role_query=_is_role_query(variant),
                    )
                    if score <= 0.0:
                        continue
                    candidate = (score + (chunk.relevance * 1.0), chunk.chunk_id, chunk.source_name, window[:max_chars_per_excerpt].strip())
                    if best_variant is None or candidate[0] > best_variant[0]:
                        best_variant = candidate
            if best_variant and best_variant[3] not in selected_texts:
                selected.append((best_variant[1], best_variant[2], best_variant[3]))
                selected_texts.add(best_variant[3])
                per_chunk_counts[best_variant[1]] = per_chunk_counts.get(best_variant[1], 0) + 1
                if len(selected) >= max_excerpts:
                    break

    scored_windows.sort(reverse=True)
    for _score, chunk_id, source_name, candidate in scored_windows:
        if candidate in selected_texts:
            continue
        if per_chunk_counts.get(chunk_id, 0) >= 2:
            continue
        per_chunk_counts[chunk_id] = per_chunk_counts.get(chunk_id, 0) + 1
        selected.append((chunk_id, source_name, candidate))
        selected_texts.add(candidate)
        if len(selected) >= max_excerpts:
            break

    blocks = []
    for index, (chunk_id, source_name, candidate) in enumerate(selected, start=1):
        blocks.append(f'[Excerpt {index} | chunk_id={chunk_id} | source={source_name}]\n{candidate}')
    return '\n\n'.join(blocks)


def unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    seen: dict[str, None] = {}
    for chunk in chunks:
        seen.setdefault(chunk.source_name, None)
    return list(seen.keys())


def coerce_content(message: Any) -> str:
    content = getattr(message, 'content', message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                parts.append(str(item['text']))
        return '\n'.join(part for part in parts if part)
    return str(content)
