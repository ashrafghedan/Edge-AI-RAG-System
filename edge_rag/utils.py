from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def normalize_question_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r'[^a-z0-9\s]', '', lowered)
    lowered = re.sub(r'\s+', ' ', lowered)
    return lowered.strip()


def safe_label(names: list[str], max_length: int = 80) -> str:
    joined = ', '.join(names)
    if len(joined) <= max_length:
        return joined
    return joined[: max_length - 3].rstrip() + '...'


def _repair_json_candidate(candidate: str) -> str:
    output: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False

    for character in candidate:
        if in_string:
            output.append(character)
            if escape:
                escape = False
            elif character == '\\':
                escape = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            output.append(character)
            continue

        if character in '{[':
            stack.append(character)
            output.append(character)
            continue

        if character in '}]':
            target = '{' if character == '}' else '['
            while stack and stack[-1] != target:
                output.append(']' if stack.pop() == '[' else '}')
            if stack and stack[-1] == target:
                stack.pop()
                output.append(character)
            continue

        output.append(character)

    while stack:
        output.append(']' if stack.pop() == '[' else '}')

    return ''.join(output)


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1:
            raise ValueError('No JSON object found in model output.')
        candidate = text[start : end + 1] if end != -1 and end > start else text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _repair_json_candidate(candidate)
        return json.loads(repaired)


def atomic_write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False, dir=path.parent) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        temp_name = handle.name
    Path(temp_name).replace(path)
