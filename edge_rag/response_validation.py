from __future__ import annotations

import re


_DANGLING_CONNECTORS = {
    'a',
    'and',
    'an',
    'as',
    'about',
    'after',
    'before',
    'because',
    'between',
    'but',
    'during',
    'for',
    'from',
    'if',
    'into',
    'of',
    'or',
    'over',
    'so',
    'than',
    'that',
    'the',
    'then',
    'to',
    'toward',
    'through',
    'under',
    'when',
    'while',
    'within',
    'without',
    'with',
}
_DANGLING_TRAILING_CHARS = {':', ';', ',', '-', '/', '\\', '(', '[', '{'}
_TERMINAL_CHARS = set('.!?)]}"\'`')
_LENGTH_DONE_REASONS = {'length', 'max_tokens', 'num_predict', 'token_limit'}
_LIST_MARKER_ONLY_RE = re.compile(r'^\s*(?:[-*+]\s*|\d+\.\s*)$')
_LIST_ITEM_RE = re.compile(r'^\s*(?:[-*+]\s+|\d+\.\s+)')
_CODEISH_LINE_RE = re.compile(r'[=:{}()[\]<>]|^\s{2,}|^[-*+]\s+`|^```')
_JSONISH_START_RE = re.compile(r'^\s*(?:```json\s*)?[\[{]', flags=re.IGNORECASE)


def response_needs_continuation(
    text: str,
    *,
    done_reason: str = '',
    expected_format: str | None = None,
) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    if str(done_reason or '').strip().lower() in _LENGTH_DONE_REASONS:
        return True
    if has_unclosed_code_fence(stripped):
        return True
    if _looks_like_structured_output(stripped, expected_format=expected_format) and _has_unbalanced_json_brackets(stripped):
        return True
    if _ends_with_dangling_list_marker(stripped):
        return True
    if _ends_with_dangling_connector(stripped):
        return True
    if _ends_with_unterminated_sentence(stripped):
        return True
    return False


def finalize_response_text(text: str, *, expected_format: str | None = None) -> str:
    finalized = text.rstrip()
    if has_unclosed_code_fence(finalized):
        finalized = finalized + '\n```'
    if _looks_like_structured_output(finalized, expected_format=expected_format):
        finalized = _close_balanced_json_tail(finalized)
    return finalized.strip()


def has_unclosed_code_fence(text: str) -> bool:
    return text.count('```') % 2 == 1


def _looks_like_structured_output(text: str, *, expected_format: str | None = None) -> bool:
    if str(expected_format or '').lower() == 'json':
        return True
    return _JSONISH_START_RE.search(text) is not None


def _ends_with_dangling_list_marker(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return _LIST_MARKER_ONLY_RE.fullmatch(lines[-1]) is not None


def _ends_with_dangling_connector(text: str) -> bool:
    tail = text.rstrip()
    if not tail:
        return False
    match = re.search(r"([a-zA-Z']+)\s*$", tail)
    if match is None:
        return False
    return match.group(1).lower() in _DANGLING_CONNECTORS


def _ends_with_unterminated_sentence(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    last_line = lines[-1].strip()
    if not last_line:
        return False
    is_list_item = False
    list_item_match = _LIST_ITEM_RE.match(last_line)
    if list_item_match:
        is_list_item = True
        last_line = last_line[list_item_match.end():].strip()
        if not last_line:
            return True
    if last_line[-1] in _TERMINAL_CHARS:
        return False
    if last_line[-1] in _DANGLING_TRAILING_CHARS:
        return True
    if is_list_item:
        return False
    if _CODEISH_LINE_RE.search(last_line):
        return False
    word_count = len(re.findall(r"[A-Za-z0-9']+", last_line))
    if word_count >= 4:
        return True
    total_word_count = len(re.findall(r"[A-Za-z0-9']+", text))
    return total_word_count >= 40 and word_count >= 2


def _has_unbalanced_json_brackets(text: str) -> bool:
    stack: list[str] = []
    in_string = False
    escape = False
    for character in text:
        if in_string:
            if escape:
                escape = False
            elif character == '\\':
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in '{[':
            stack.append(character)
            continue
        if character == '}' and stack and stack[-1] == '{':
            stack.pop()
            continue
        if character == ']' and stack and stack[-1] == '[':
            stack.pop()
            continue
    return in_string or bool(stack)


def _close_balanced_json_tail(text: str) -> str:
    if not text or _has_unbalanced_codeish_string(text):
        return text

    stack: list[str] = []
    in_string = False
    escape = False
    for character in text:
        if in_string:
            if escape:
                escape = False
            elif character == '\\':
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in '{[':
            stack.append(character)
            continue
        if character == '}' and stack and stack[-1] == '{':
            stack.pop()
            continue
        if character == ']' and stack and stack[-1] == '[':
            stack.pop()
            continue

    if in_string:
        return text

    closers = ''.join('}' if opener == '{' else ']' for opener in reversed(stack))
    return text + closers


def _has_unbalanced_codeish_string(text: str) -> bool:
    escaped = False
    in_string = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
    return in_string
