"""PDF -> RAG-ready text + JSONL conversion.

Pipeline (each page):
    1. Block-based digital extraction (preserves reading order via y,x sort).
    2. Quality scoring: digital extraction is rejected if the result looks
       garbled (replacement chars, low alpha-ratio, runs of tiny lines,
       repeated short tokens). Bad pages fall back to Tesseract OCR.
    3. Cleanup pass: unicode normalization, hyphen-break repair, drop-cap
       de-duplication, joined-preposition splits, missing-space fixes,
       textbook noise stripping (photo credits, map coords, chapter labels).
    4. Cross-page running-header detection (drops lines that appear on a
       large fraction of pages -- chapter banners, footers, etc.).
    5. Section tracking: title-cased / ALL CAPS short lines are recorded as
       the current section; carried forward across pages.
    6. Content classification: review questions, captions, map/textbox
       sidebars are prefixed in-line so RAG chunks can be filtered.
    7. Optional visual description via the local llama.cpp VLM
       (LLAMA_CPP_*); off by default. Embedded figures are tagged
       ``[Figure Description] ...`` and page screenshots are tagged
       ``[Page Visual Description] ...``.

Outputs:
    ``<pdf>_extracted.txt``   - human-readable, ``--- Page N ---`` markers
    ``<pdf>_extracted.jsonl`` - one row per page with text + visual fields

Everything runs locally (PyMuPDF + Tesseract + an optional local VLM).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from tqdm import tqdm

from edge_rag.env import load_project_env


# ── Project paths ───────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
load_project_env(PROJECT_ROOT)
TESSDATA_DIR = PROJECT_ROOT / 'data' / 'tessdata'
if TESSDATA_DIR.exists():
    os.environ.setdefault('TESSDATA_PREFIX', str(TESSDATA_DIR))


# ── Environment-driven config (kept for backwards compatibility) ────────

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


LLAMA_BASE_URL = (
    os.environ.get('LLAMA_CPP_BASE_URL')
    or os.environ.get('OLLAMA_BASE_URL')
    or 'http://127.0.0.1:11436'
).rstrip('/')

VISION_MODEL = (
    os.environ.get('LLAMA_CPP_MODEL')
    or os.environ.get('EDGE_RAG_ANSWER_MODEL')
    or 'gemma-4-e2b-q4km'
)

REQUEST_TIMEOUT = int(
    os.environ.get('LLAMA_CPP_REQUEST_TIMEOUT')
    or os.environ.get('EDGE_RAG_OLLAMA_REQUEST_TIMEOUT')
    or '300'
)

PDF_DPI = int(os.environ.get('CNV_PDF_DPI', '200'))
PAGE_TEXT_THRESHOLD = int(os.environ.get('CNV_PDF_TEXT_THRESHOLD', '60'))
IMAGE_PAGE_AREA_LIMIT = float(os.environ.get('CNV_PDF_IMAGE_MIN_RATIO', '0.05'))
OCR_LANG = os.environ.get('CNV_PDF_OCR_LANG', 'eng')
MAX_FIGURES = int(os.environ.get('CNV_PDF_MAX_FIGURES', '0'))
DESCRIBE_PAGES = _env_flag('CNV_PDF_DESCRIBE_PAGES', False)
FORCE_PAGE_ANALYSIS = _env_flag('CNV_PDF_FORCE_PAGE_ANALYSIS', False)
VLM_MAX_IMAGE_SIDE = int(os.environ.get('CNV_PDF_VLM_MAX_IMAGE_SIDE', '512'))
VLM_PAGE_MAX_IMAGE_SIDE = int(os.environ.get('CNV_PDF_VLM_PAGE_MAX_IMAGE_SIDE', '384'))
VLM_PAGE_MAX_TOKENS = max(320, int(os.environ.get('CNV_PDF_VLM_PAGE_MAX_TOKENS', '320')))
VLM_FIGURE_MAX_TOKENS = int(os.environ.get('CNV_PDF_VLM_FIGURE_MAX_TOKENS', '384'))
PAGE_RENDER_SCALE = float(os.environ.get('CNV_PDF_PAGE_RENDER_SCALE', '2.0'))
VISUAL_TEXT_RICH_THRESHOLD = int(os.environ.get('CNV_PDF_VISUAL_TEXT_RICH_THRESHOLD', '1400'))
TEXT_RICH_MAX_FIGURES = int(os.environ.get('CNV_PDF_TEXT_RICH_MAX_FIGURES', '1'))
TEXT_RICH_IMAGE_SIDE = int(os.environ.get('CNV_PDF_TEXT_RICH_IMAGE_SIDE', '192'))
TEXT_RICH_FIGURE_MAX_TOKENS = int(os.environ.get('CNV_PDF_TEXT_RICH_FIGURE_MAX_TOKENS', '64'))

# Block extraction: drop sub-2-char blocks (decorative pixels), and snap y0
# to a small grid so block-sort doesn't shuffle baseline-aligned lines.
MIN_BLOCK_TEXT_LEN = 2
BLOCK_Y_BIN = 4.0


# ═══════════════════════════════════════════════════════════════════════
# Page quality scoring
# ═══════════════════════════════════════════════════════════════════════

def _alphabetic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def is_bad_extraction(text: str) -> bool:
    """Heuristic: should we re-OCR this page even if it has some text?

    Triggers when the digital extraction looks broken: replacement chars,
    low letter-ratio, lots of tiny lines, or one short token spammed.
    Conservative on purpose -- false negatives are cheaper than re-OCRing
    every page of a clean digital PDF.
    """
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 30:
        return True

    # Replacement / soft-hyphen flood = font fallback failure
    if stripped.count('�') > 3 or stripped.count('­') > 8:
        return True

    # Mostly punctuation/digits = decorative page or extraction garbage
    if _alphabetic_ratio(stripped) < 0.4:
        return True

    # Lots of tiny lines = chunked extraction; usually a layout-heavy page
    lines = [ln for ln in stripped.split('\n') if ln.strip()]
    if len(lines) >= 8:
        tiny = sum(1 for ln in lines if len(ln.strip()) <= 2)
        if tiny / len(lines) > 0.35:
            return True

    # Same short token repeated dozens of times = broken extraction loop
    short_words = [w for w in re.findall(r'\b\w+\b', stripped) if 1 <= len(w) <= 3]
    if len(short_words) >= 20:
        most_common, n = Counter(short_words).most_common(1)[0]
        if n > 12 and n / len(short_words) > 0.5:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# Digital text extraction (block-sorted)
# ═══════════════════════════════════════════════════════════════════════

def _read_page_blocks(page: fitz.Page) -> str:
    """Pull text blocks and rejoin them in natural reading order.

    PyMuPDF's ``page.get_text('text')`` gives a single flat string in PDF
    object order, which is often *not* reading order. ``blocks`` exposes
    bounding boxes, so we can sort by y (top-to-bottom) then x.
    """
    blocks = page.get_text('blocks') or []
    items: list[tuple[float, float, str]] = []
    for b in blocks:
        if len(b) < 7:
            continue
        x0, y0, _x1, _y1, text, _no, btype = b[:7]
        if btype != 0:                                   # 0 = text, 1 = image
            continue
        snippet = (text or '').strip()
        if len(snippet) < MIN_BLOCK_TEXT_LEN:
            continue
        # Snap y0 to a coarse grid so blocks on the same baseline don't
        # flip-flop due to sub-point coordinate noise.
        items.append((round(y0 / BLOCK_Y_BIN) * BLOCK_Y_BIN, x0, snippet))
    items.sort(key=lambda t: (t[0], t[1]))
    return '\n\n'.join(item[2] for item in items).strip()


# ═══════════════════════════════════════════════════════════════════════
# Tesseract OCR (via pymupdf)
# ═══════════════════════════════════════════════════════════════════════

def _ocr_page(page: fitz.Page, *, lang: str, dpi: int) -> str:
    try:
        tp = page.get_textpage_ocr(
            language=lang,
            dpi=dpi,
            full=True,
            tessdata=str(TESSDATA_DIR) if TESSDATA_DIR.exists() else None,
        )
        return page.get_text(textpage=tp) or ''
    except Exception as exc:
        print(f' [!] OCR failed on page {page.number + 1}: {exc}', file=sys.stderr)
        return ''


# ═══════════════════════════════════════════════════════════════════════
# llama.cpp VLM (optional figure descriptions)
# ═══════════════════════════════════════════════════════════════════════

def _visual_prompt(kind: str) -> str:
    if kind == 'page':
        return (
            'Read this textbook page image directly. '
            'Write one short paragraph, about 3 to 5 sentences. '
            'Include the page title or section heading, the main visual elements, '
            'important labels or captions, and the key relationships shown. '
            'Mention readable text only when it matters. '
            'Do not invent details, do not explain your process, and do not use bullet points.'
        )
    return (
        'Describe this textbook figure or panel in 2 to 4 sentences. '
        'Mention readable labels, captions, legends, axes, arrows, and the main relationship shown. '
        'Do not invent details or explain your process.'
    )


_VLM_REFUSAL_RE = re.compile(
    r"(i(?: am|'m) sorry.*?(?:can't|cannot|do not|don't)|"
    r"do not have (?:the )?ability to view|"
    r"don't have (?:the )?ability to view|"
    r"don't have access to the image|"
    r"cannot provide a transcription|"
    r"can't provide a transcription|"
    r"cannot view the image|"
    r"can't view the image|"
    r"cannot interpret visual content|"
    r"can't interpret visual content)",
    re.IGNORECASE | re.DOTALL,
)

_VLM_REASONING_STEP_RE = re.compile(
    r'^\d+\.\s+\*{0,2}(analyze|examine|identify|interpret|plan|request|understand)\b',
    re.IGNORECASE,
)

_VLM_DESCRIPTION_MARKER_RE = re.compile(
    r'(Detailed Transcription(?: and)? Description(?:\s*\([^)]*\))?|'
    r'Detailed Description(?:\s*\([^)]*\))?|'
    r'Final Description(?:\s*\([^)]*\))?|'
    r'Transcription(?: and| &)? Description(?:\s*\([^)]*\))?)\s*:?',
    re.IGNORECASE,
)

_VLM_REASONING_NOISE_RE = re.compile(
    r'(thinking process|analyze the request|examine the image|initial scan\s*&\s*context|section by section)',
    re.IGNORECASE,
)

_VLM_REASONING_BULLET_RE = re.compile(r'^(?:[-*]\s*)+')

_VLM_TITLE_LABEL_RE = re.compile(
    r'^(?:Title/Main Heading|Main Heading|Title)\s*:\s*',
    re.IGNORECASE,
)

_VLM_PROMPT_ECHO_RE = re.compile(
    r'^(?:Input:\s*An image.*?Constraint:\s*Do not invent details\.?\s*Keep it concise but useful for retrieval\.?\s*)',
    re.IGNORECASE,
)

_VLM_CHECKLIST_ECHO_RE = re.compile(
    r'^(?:Must include:.*?Do not use bullet points\?\s*Yes\.\s*)',
    re.IGNORECASE,
)


def _looks_like_vlm_refusal(text: str) -> bool:
    compact = re.sub(r'\s+', ' ', text or '').strip()
    if not compact:
        return False
    return bool(_VLM_REFUSAL_RE.search(compact))


def _strip_vlm_prompt_echo(text: str) -> str:
    compact = re.sub(r'\s+', ' ', text or '').strip()
    if not compact:
        return ''
    compact = _VLM_PROMPT_ECHO_RE.sub('', compact).strip()
    compact = _VLM_CHECKLIST_ECHO_RE.sub('', compact).strip()
    marker_match = re.search(r'(?:The|This) image\b', compact, re.IGNORECASE)
    if marker_match and marker_match.start() > 0:
        prefix = compact[:marker_match.start()].strip()
        if any(
            token in prefix.lower()
            for token in (
                'task 1', 'task 2', 'task 3', 'constraint', 'input:', 'transcription:', 'description:',
                'must include:', 'constraint checklist:', 'write one short paragraph', '3 to 5 sentences',
            )
        ):
            compact = compact[marker_match.start():].strip()
    return compact


def _sanitize_vlm_reasoning_fallback(text: str) -> str:
    original = (text or '').replace('\r', '').strip()
    if not original:
        return ''
    compact = original
    compact = re.sub(r'^\s*thinking process:\s*', '', compact, flags=re.IGNORECASE)
    compact = re.sub(
        r'^\s*here(?:\'s| is)\s+a\s+thinking\s+process.*?:\s*',
        '',
        compact,
        flags=re.IGNORECASE,
    )
    marker_match = _VLM_DESCRIPTION_MARKER_RE.search(compact)
    if marker_match:
        compact = compact[marker_match.end():].lstrip(' :*-')
    lines: list[str] = []
    for raw_line in compact.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = _VLM_REASONING_BULLET_RE.sub('', stripped)
        stripped = stripped.replace('**', '').strip()
        stripped = re.sub(r'^\((?:Section by Section|Initial Scan & Context)\)\s*:\s*', '', stripped, flags=re.IGNORECASE)
        stripped = _VLM_TITLE_LABEL_RE.sub('Title: ', stripped)
        if _VLM_REASONING_STEP_RE.match(stripped):
            continue
        if _VLM_REASONING_NOISE_RE.search(stripped):
            continue
        lines.append(stripped)
    compact = ' '.join(lines)
    compact = re.sub(r'\s+', ' ', compact).strip(' :-')
    if len(compact) < 48 or _VLM_REASONING_NOISE_RE.search(compact):
        fallback_bits: list[str] = []
        normalized_original = re.sub(r'\s+', ' ', original).strip()
        title_match = re.search(
            r'(?:Title/Main Heading|Main Heading|Title)\s*:\s*["“]?([^"“”*]+?)["”]?(?=\s*(?:\*|$))',
            normalized_original,
            re.IGNORECASE,
        )
        if title_match:
            title = title_match.group(1).strip(' .:-')
            if title:
                fallback_bits.append(f'Title: {title}.')
        image_match = re.search(
            r'(?:^|[.!?]\s+|\*\s+)((?:The|This) image .*?)(?=(?:\s+\d+\.\s)|(?:\s+\*\s+\*\*)|$)',
            normalized_original,
            re.IGNORECASE,
        )
        if image_match:
            sentence = image_match.group(1).strip(' :-')
            if sentence:
                fallback_bits.append(sentence.rstrip('.') + '.')
        if fallback_bits:
            compact = ' '.join(fallback_bits).strip()
    if len(compact) < 48:
        match = re.search(r'((?:The|This) image .*?)(?=(?:\d+\.\s)|$)', normalized_original, re.IGNORECASE)
        if match:
            compact = match.group(1).strip(' :-')
    return compact


def _post_vlm(
    prompt: str,
    image_bytes: bytes,
    *,
    max_tokens: int,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    context: str | None = None,
    warn: bool = True,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    eff_model = model or VISION_MODEL
    eff_base_url = (base_url or LLAMA_BASE_URL).rstrip('/')
    eff_timeout = REQUEST_TIMEOUT if timeout is None else int(timeout)
    encoded = base64.b64encode(image_bytes).decode('ascii')
    payload = {
        'model': eff_model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{encoded}'}},
            ],
        }],
        'temperature': 0.1,
        'max_tokens': int(max_tokens),
        'stream': False,
    }
    request = urllib.request.Request(
        f'{eff_base_url}/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    def warn_failure(message: str) -> None:
        if diagnostics is not None:
            diagnostics['vlm_failures'] = int(diagnostics.get('vlm_failures', 0)) + 1
        if not warn:
            return
        ctx = f' ({context})' if context else ''
        print(
            f' [!] VLM warning {eff_base_url}/v1/chat/completions{ctx}: {message}',
            file=sys.stderr,
        )

    try:
        with urllib.request.urlopen(request, timeout=eff_timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = f'HTTPError {exc.code}: {exc.reason}'
        try:
            payload = exc.read().decode('utf-8', errors='replace').strip()
        except Exception:
            payload = ''
        if payload:
            detail = f'{detail} | {payload}'
        warn_failure(detail)
        return ''
    except Exception as exc:
        warn_failure(f'{type(exc).__name__}: {exc}')
        return ''

    choice = (body.get('choices') or [{}])[0]
    message = choice.get('message') if isinstance(choice, dict) else None
    content = (message or {}).get('content') or ''
    reasoning_content = (message or {}).get('reasoning_content') or ''
    if isinstance(content, list):
        content = ''.join(
            str(item.get('text') or '') if isinstance(item, dict) else str(item)
            for item in content
        )
    if isinstance(reasoning_content, list):
        reasoning_content = ''.join(
            str(item.get('text') or '') if isinstance(item, dict) else str(item)
            for item in reasoning_content
        )
    text = str(content).strip()
    if not text:
        text = _sanitize_vlm_reasoning_fallback(str(reasoning_content))
    text = _strip_vlm_prompt_echo(text)
    if not text:
        warn_failure('empty response')
        return ''
    if _looks_like_vlm_refusal(text):
        warn_failure('refusal-style response')
        return ''
    return text


def describe_image(
    image_bytes: bytes,
    *,
    prompt: str | None = None,
    max_tokens: int = VLM_FIGURE_MAX_TOKENS,
    context: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    eff_prompt = prompt or _visual_prompt('figure')
    text = _post_vlm(
        eff_prompt,
        image_bytes,
        max_tokens=max_tokens,
        model=model,
        base_url=base_url,
        timeout=timeout,
        context=context,
        diagnostics=diagnostics,
    )
    return re.sub(r'\s+', ' ', text).strip()


def _shrink_pixmap_for_vlm(pix: fitz.Pixmap, *, max_side: int = VLM_MAX_IMAGE_SIDE) -> fitz.Pixmap:
    if max_side <= 0:
        return pix
    while max(pix.width, pix.height) > max_side:
        pix.shrink(1)
    return pix


def _extract_page_images(
    page: fitz.Page,
    doc: fitz.Document,
    *,
    max_side: int = VLM_MAX_IMAGE_SIDE,
) -> list[bytes]:
    """Return PNG bytes for embedded images large enough to be informative."""
    page_area = float(page.rect.width * page.rect.height) or 1.0
    candidates: list[tuple[float, bytes]] = []
    seen: set[int] = set()
    for img_meta in page.get_images(full=True):
        xref = img_meta[0]
        if xref in seen:
            continue
        seen.add(xref)
        area_ratio = 0.0
        try:
            rects = list(page.get_image_rects(xref))
        except Exception:
            rects = []
        if rects:
            largest = max(rects, key=lambda r: r.width * r.height)
            area_ratio = (largest.width * largest.height) / page_area
            if area_ratio < IMAGE_PAGE_AREA_LIMIT:
                continue
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            if pix.colorspace and pix.colorspace.n > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix = _shrink_pixmap_for_vlm(pix, max_side=max_side)
            candidates.append((area_ratio, pix.tobytes('png')))
        except Exception:
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates]


_VISUAL_KEYWORD_RE = re.compile(
    r'\b(map|chart|diagram|timeline|table|figure|comparing|key|legend|graph)\b',
    re.IGNORECASE,
)


def _render_page_png(
    page: fitz.Page,
    *,
    scale: float = PAGE_RENDER_SCALE,
    max_side: int = VLM_PAGE_MAX_IMAGE_SIDE,
) -> bytes:
    page_max_dim = max(float(page.rect.width), float(page.rect.height), 1.0)
    eff_scale = max(scale, 0.1)
    if max_side > 0:
        eff_scale = min(eff_scale, max_side / page_max_dim)
        eff_scale = max(eff_scale, 0.1)
    pix = page.get_pixmap(matrix=fitz.Matrix(eff_scale, eff_scale), alpha=False)
    pix = _shrink_pixmap_for_vlm(pix)
    return pix.tobytes('png')


def _describe_page_with_retries(
    page: fitz.Page,
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    attempts = [
        (VLM_PAGE_MAX_IMAGE_SIDE, VLM_PAGE_MAX_TOKENS),
        (min(VLM_PAGE_MAX_IMAGE_SIDE, 384), min(VLM_PAGE_MAX_TOKENS, 320)),
        (256, 224),
    ]
    seen: set[tuple[int, int]] = set()
    ordered_attempts: list[tuple[int, int]] = []
    for max_side, max_tokens in attempts:
        key = (max(128, int(max_side)), max(128, int(max_tokens)))
        if key in seen:
            continue
        seen.add(key)
        ordered_attempts.append(key)

    last_desc = ''
    for index, (max_side, max_tokens) in enumerate(ordered_attempts):
        if index > 0:
            try:
                _ensure_vlm_ready(
                    model=model,
                    base_url=base_url,
                    timeout=max(20, min(int(timeout or REQUEST_TIMEOUT), 90)),
                )
            except RuntimeError:
                pass
        try:
            last_desc = describe_image(
                _render_page_png(page, max_side=max_side),
                prompt=_visual_prompt('page'),
                max_tokens=max_tokens,
                context=f'page {page.number + 1} screenshot',
                model=model,
                base_url=base_url,
                timeout=timeout,
                diagnostics=diagnostics,
            )
        except Exception:
            last_desc = ''
        if last_desc:
            return last_desc
    return last_desc


def _fetch_vlm_models(
    *,
    base_url: str | None = None,
    timeout: int = 3,
) -> list[str]:
    eff_base_url = (base_url or LLAMA_BASE_URL).rstrip('/')
    request = urllib.request.Request(f'{eff_base_url}/v1/models', method='GET')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode('utf-8'))
    models = payload.get('data') if isinstance(payload, dict) else []
    if not isinstance(models, list):
        return []
    result: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get('id') or '').strip()
        if model_id:
            result.append(model_id)
    return result


def _start_local_vlm_helper() -> subprocess.Popen[bytes] | None:
    helper = PROJECT_ROOT / 'scripts' / 'run_llama_cpp.js'
    if not helper.exists():
        return None

    candidates = ['node']
    for command in candidates:
        try:
            return subprocess.Popen(
                [command, str(helper)],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            continue
        except Exception:
            return None
    return None


def _ensure_vlm_ready(
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int = 90,
) -> None:
    eff_model = (model or VISION_MODEL).strip()
    eff_base_url = (base_url or LLAMA_BASE_URL).rstrip('/')
    try:
        loaded = _fetch_vlm_models(base_url=eff_base_url, timeout=3)
        if not eff_model or eff_model in loaded:
            return
        raise RuntimeError(
            f'VLM server at {eff_base_url} is up, but {eff_model} is not loaded. '
            f'Loaded models: {", ".join(loaded) or "(none reported)"}'
        )
    except RuntimeError:
        raise
    except Exception:
        pass

    print(f'  VLM      : starting local llama.cpp server at {eff_base_url} ...')
    helper_process = _start_local_vlm_helper()
    if helper_process is None:
        raise RuntimeError(
            'Could not auto-start llama.cpp. Run `npm run dev:llama` in another terminal first.'
        )

    deadline = time.monotonic() + max(5, int(timeout))
    last_error = 'VLM server did not become ready.'
    while time.monotonic() < deadline:
        if helper_process.poll() not in (None, 0):
            raise RuntimeError(
                'The local llama.cpp helper exited before the VLM server became ready. '
                'Run `npm run dev:llama` manually and try again.'
            )
        try:
            loaded = _fetch_vlm_models(base_url=eff_base_url, timeout=3)
            if not eff_model or eff_model in loaded:
                print(f'  VLM      : ready ({eff_model} @ {eff_base_url})')
                return
            last_error = (
                f'VLM server became reachable, but {eff_model} is not loaded. '
                f'Loaded models: {", ".join(loaded) or "(none reported)"}'
            )
            break
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.75)

    raise RuntimeError(
        f'Could not reach the local VLM server at {eff_base_url}. '
        f'Last error: {last_error}'
    )


def _visual_page_stats(page: fitz.Page) -> tuple[int, float, float]:
    page_area = float(page.rect.width * page.rect.height) or 1.0
    seen: set[int] = set()
    ratios: list[float] = []
    for img_meta in page.get_images(full=True):
        xref = img_meta[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            rects = list(page.get_image_rects(xref))
        except Exception:
            rects = []
        if not rects:
            continue
        largest = max(rects, key=lambda r: r.width * r.height)
        ratios.append((largest.width * largest.height) / page_area)
    if not ratios:
        return 0, 0.0, 0.0
    return len(ratios), max(ratios), sum(ratios)


def looks_visual_heavy(page: fitz.Page, text: str) -> bool:
    """Heuristic for textbook pages where a page screenshot is more useful."""
    compact_text = re.sub(r'\s+', ' ', text or '').strip()
    if len(compact_text) >= VISUAL_TEXT_RICH_THRESHOLD:
        return False
    embedded_images, largest_ratio, total_ratio = _visual_page_stats(page)
    text_is_short = len(compact_text) < PAGE_TEXT_THRESHOLD
    has_visual_keyword = bool(_VISUAL_KEYWORD_RE.search(compact_text))
    image_dominant = largest_ratio >= 0.35 or total_ratio >= 0.60
    return text_is_short or image_dominant or (has_visual_keyword and embedded_images > 0)


# ═══════════════════════════════════════════════════════════════════════
# Cleanup pipeline (regex-only -- adds < 100ms even on big books)
# ═══════════════════════════════════════════════════════════════════════

# Generic cleanup
_URL_RE = re.compile(r'https?://\S+')
_MULTI_BLANK_RE = re.compile(r'\n{3,}')

# Smart-quote -> ASCII, plus drop replacement and soft-hyphen artifacts
_UNICODE_NORMALIZE = str.maketrans({
    '‘': "'", '’': "'",
    '“': '"', '”': '"',
    '–': '-', '—': '-',
    '\xa0': ' ',
    '�': '', '￾': '',
    '­': '',
})

# Word continued onto the next line: "satis-\nfaction" -> "satisfaction"
_HYPHEN_LINE_BREAK_RE = re.compile(r'([a-zA-Z])[-~­]\s*\n\s*([a-z])')
# Tilde mid-word OCR artifact: "satis~ faction" -> "satisfaction"
_TILDE_BREAK_RE = re.compile(r'([a-z])~\s*([a-z])')

# Joined preposition + closed-class word (no real English word looks like this)
_PREP_FIX_RE = re.compile(
    r'\b(of|in|on|at|by|to|for|with|from|and|or)'
    r'(the|its|all|any|each|every|some|many|both|few|most|other|'
    r'this|that|these|those|our|your|their|his|her)\b',
    re.IGNORECASE,
)
# Same, but for domain-flavoured words common in textbook OCR
_DOMAIN_FIX_RE = re.compile(
    r'\b(of|in|on|by|to|for|with|from)'
    r'(reproduction|photosynthesis|bacteria|protists|fungi|plants|animals|'
    r'cells|species|life|body|nature|example|examples|methods|members|'
    r'study|water|oxygen|carbon|hydrogen|nitrogen|chapter|chapters|'
    r'organisms|classification|biology|chemistry|physics|history|government|'
    r'religion|society|culture|economy|empire|civilization)\b',
    re.IGNORECASE,
)
# Preposition glued to a CamelCase word -- always safe to split
_PREP_CAMEL_RE = re.compile(
    r'\b(of|in|on|at|by|to|for|with|from|and|or)([A-Z][a-z]+)',
)
_QUANTITY_OF_RE = re.compile(
    r'\b(one|two|three|four|five|six|seven|eight|nine|ten|each|many|some|'
    r'several|both|few|most|all|half|kind|kinds|type|types|sort|sorts|pair|'
    r'part|parts|members|cells|methods|examples|study|studies|species|set|sets|'
    r'group|groups|list|number|numbers|amount|amounts|series|range|piece|pieces)'
    r'(of)\b',
    re.IGNORECASE,
)
_TRAILING_OF_RE = re.compile(
    r'\b([a-zA-Z]{3,})(of)\b(?=\s+(the|a|an|its|each|some|all|this|that|these|those))',
)

# Drop-cap leftover INSIDE a line: "Early arly" -> "Early"
_INLINE_DROP_CAP_RE = re.compile(r'\b([A-Z][a-z]+)\s+([a-z]+)\b')
# Drop-cap leftover ACROSS lines: "Early\narly" -> "Early"
_LINE_DROP_CAP_RE = re.compile(r'\b([A-Z][a-zA-Z]{3,})\s*\n\s*([a-z][a-zA-Z]{2,})\b')

# Missing space after sentence-ending punctuation: "word.Other" -> "word. Other".
# Requires [A-Z][a-z] after, so "U.S.A." and ".com" survive unchanged.
_PUNCT_SPACE_RE = re.compile(r'([.!?])([A-Z][a-z])')
# Missing space after a comma in front of a letter (numbers like 1,000 are safe).
_COMMA_SPACE_RE = re.compile(r',([a-zA-Z])')

# Pure-junk lines we always want gone
_NOISE_LINE_RE = re.compile(r'^[\W_]+$|^\d{1,4}$')
_SHORT_FRAGMENT_RE = re.compile(r'^[a-zA-Z0-9]{1,2}$')
_DECORATIVE_RE = re.compile(r'^[|{}()\[\]<>/\\!?*+=\-_~`^"\'.,;:\s]{0,4}$')

# Map coordinate axis labels: "60°E", "0°", "30°N"
_GEO_COORD_RE = re.compile(r'^\s*-?\d+(\.\d+)?\s*°\s*[NSEW]?\s*$', re.IGNORECASE)
# Letter-spaced toponyms: "A F R I C A" -> "AFRICA"
_LETTER_SPACED_RE = re.compile(r'^[A-Z](\s+[A-Z]){1,12}$')


# Textbook noise patterns -----------------------------------------------

# Stand-alone noise lines often produced by Glencoe/Pearson/etc. layouts.
_TEXTBOOK_NOISE_PATTERNS = [
    re.compile(r'^Page\s+\d+$', re.IGNORECASE),
    re.compile(r'^CONTENTS$'),
    re.compile(r'^TABLE\s+OF\s+CONTENTS$', re.IGNORECASE),
    # Print-shop production codes like "008-015 C1S1 ..." or "002 FM ..."
    re.compile(r'^\d{3,4}-\d{3,4}\s+[A-Z]\w*'),
    re.compile(r'^\d{3,4}\s+[A-Z]\d[A-Z]\d'),
    # "Visit jat.glencoe.com" / "Visit www.mcgraw-hill.com"
    re.compile(r'\bvisit\s+(?:www\.|https?://)?\S+\.(?:com|org|edu)\b', re.IGNORECASE),
    # Photo credit leaders: "(t)Name", "(t to b)Name", "(l to r)Name"
    re.compile(r'^\([trblcm](?:\s+to\s+[trblcm])?\)[A-Z]'),
    # Copyright lines
    re.compile(r'^©', ),
    re.compile(r'^Copyright\s+©', re.IGNORECASE),
    re.compile(r'^All\s+rights\s+reserved\.?$', re.IGNORECASE),
    # Cross-references: "Chapter 2, page 66" / "Chapter 1, page 12"
    re.compile(r'^Chapter\s+\d+,?\s*page\s*\d+\.?$', re.IGNORECASE),
]

# Strings that nearly always indicate a credit/source line
_CREDIT_INDICATORS = (
    'CORBIS', 'Getty Images', 'Art Resource',
    'Photo Researchers', 'National Geographic',
    'SuperStock', 'Bridgeman', 'Magnum', 'Sygma',
    'Worldsat', 'AP Photo', 'Mary Evans Picture Library',
    'Hulton Archive', 'Bettmann',
)


def _is_photo_credit(line: str) -> bool:
    """Detect a line that's part of a photo/source attribution block."""
    return any(ind in line for ind in _CREDIT_INDICATORS)


def _is_textbook_noise(
    line: str,
    running_headers: set[str] | None = None,
) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith('[Figure Description]') or s.startswith('[Page Visual Description]'):
        return False
    if _GEO_COORD_RE.match(s):
        return True
    for pat in _TEXTBOOK_NOISE_PATTERNS:
        if pat.search(s):
            return True
    if _is_photo_credit(s):
        return True
    if running_headers and s in running_headers:
        return True
    return False


def _strip_textbook_noise(text: str, running_headers: set[str] | None) -> str:
    """Drop lines matching the textbook-noise filters."""
    return '\n'.join(
        ln for ln in text.split('\n')
        if not _is_textbook_noise(ln, running_headers)
    )


# Drop-cap dedup helpers ------------------------------------------------

def _fix_inline_drop_cap(text: str) -> str:
    """'Early arly' -> 'Early', 'Humans umans' -> 'Humans'."""
    def repl(m: re.Match) -> str:
        w1, w2 = m.group(1), m.group(2)
        if len(w1) >= 4 and w1[1:].lower() == w2.lower():
            return w1
        return m.group(0)
    return _INLINE_DROP_CAP_RE.sub(repl, text)


def _fix_linebreak_drop_cap(text: str) -> str:
    """'Civilizations\\nivilizations' -> 'Civilizations'.

    Matches a TitleCase word followed (across a line break) by a lowercase
    fragment that's a suffix of the first word -- i.e. the drop-cap big
    letter rendered as a separate line in OCR/PDF extraction.
    """
    def repl(m: re.Match) -> str:
        w1, w2 = m.group(1), m.group(2)
        if w1.lower().endswith(w2.lower()) and len(w2) < len(w1):
            return w1
        return m.group(0)
    return _LINE_DROP_CAP_RE.sub(repl, text)


def _collapse_consecutive_duplicates(text: str) -> str:
    """Drop a line if it's identical (after strip) to the previous non-blank.

    Catches noise like ``Chapter 2 / Chapter 2`` and the repeated date axes
    on textbook timeline spreads.
    """
    out: list[str] = []
    prev_stripped: str | None = None
    for line in text.split('\n'):
        s = line.strip()
        if s and s == prev_stripped:
            continue
        out.append(line)
        prev_stripped = s if s else None
    return '\n'.join(out)


def _merge_letter_spaced_lines(text: str) -> str:
    """'A F R I C A' -> 'AFRICA' (letter-spaced toponym labels on maps)."""
    out: list[str] = []
    for line in text.split('\n'):
        s = line.strip()
        if _LETTER_SPACED_RE.fullmatch(s):
            out.append(s.replace(' ', ''))
        else:
            out.append(line)
    return '\n'.join(out)


# Master cleanup --------------------------------------------------------

def _clean_text(text: str) -> str:
    """Minimal cleanup applied right after extraction (before quality scoring)."""
    text = _URL_RE.sub('', text)
    text = _MULTI_BLANK_RE.sub('\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _post_ocr_cleanup(text: str) -> str:
    """Heavy regex-only cleanup. Idempotent; safe to call twice."""
    if not text:
        return text

    # Unicode normalization first so downstream regexes only see ASCII-ish text.
    text = text.translate(_UNICODE_NORMALIZE)

    # Glue words split across line breaks before anything else mangles them.
    text = _HYPHEN_LINE_BREAK_RE.sub(r'\1\2', text)
    text = _TILDE_BREAK_RE.sub(r'\1\2', text)
    text = _fix_linebreak_drop_cap(text)

    # Split joined prepositions (safe whitelists -- see regex defs above).
    text = _PREP_FIX_RE.sub(r'\1 \2', text)
    text = _DOMAIN_FIX_RE.sub(r'\1 \2', text)
    text = _PREP_CAMEL_RE.sub(r'\1 \2', text)
    text = _QUANTITY_OF_RE.sub(r'\1 \2', text)
    text = _TRAILING_OF_RE.sub(r'\1 \2', text)

    # In-line drop-cap leftovers: "Early arly" -> "Early"
    text = _fix_inline_drop_cap(text)

    # Restore missing spaces around punctuation.
    text = _PUNCT_SPACE_RE.sub(r'\1 \2', text)
    text = _COMMA_SPACE_RE.sub(r', \1', text)

    # Toponym letter-spacing reflow + consecutive dupes ("Chapter 2 / Chapter 2").
    text = _merge_letter_spaced_lines(text)
    text = _collapse_consecutive_duplicates(text)

    # Filter pure-noise lines (punctuation only, standalone digits, fragments).
    kept: list[str] = []
    for raw in text.split('\n'):
        s = raw.strip()
        if not s:
            kept.append('')
            continue
        if _NOISE_LINE_RE.match(s):
            continue
        if _SHORT_FRAGMENT_RE.match(s):
            continue
        if _DECORATIVE_RE.match(s):
            continue
        kept.append(raw)
    text = '\n'.join(kept)

    text = _MULTI_BLANK_RE.sub('\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════
# Cross-page analysis: running header / footer detection
# ═══════════════════════════════════════════════════════════════════════

def _detect_running_headers(
    pages_text: list[str],
    min_repeats_ratio: float = 0.3,
    min_pages: int = 4,
) -> set[str]:
    """Identify short lines that appear on a large fraction of pages.

    These are running headers/footers (chapter banner, footer label, etc.) --
    visually useful but pure noise for RAG retrieval.
    """
    total = max(1, len(pages_text))
    if total < min_pages:
        return set()                                     # too few pages to trust
    counts: Counter[str] = Counter()
    for txt in pages_text:
        seen: set[str] = set()
        for raw in (txt or '').split('\n'):
            s = raw.strip()
            if 3 <= len(s) <= 60 and not s.endswith(('.', '?', '!')):
                seen.add(s)
        counts.update(seen)
    threshold = max(2, int(total * min_repeats_ratio))
    return {line for line, n in counts.items() if n >= threshold}


# ═══════════════════════════════════════════════════════════════════════
# Section heading detection + line classification
# ═══════════════════════════════════════════════════════════════════════

_LOWER_FILLERS = {
    'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 'of', 'on',
    'or', 'the', 'to', 'with',
}


def _looks_like_heading(line: str) -> bool:
    """Conservative test for "this line looks like a section title".

    Keep this strict -- a false positive corrupts the JSONL section column
    for many downstream rows. We require:
      - 1-8 tokens, total length <= 64 chars
      - no sentence-ending punctuation
      - either ALL CAPS, or Title Case (most non-filler tokens capitalized)
    """
    s = line.strip()
    if not s or len(s) > 64:
        return False
    if s.endswith(('.', ',', ';', '?', '"', "'")):
        return False
    words = re.findall(r"[A-Za-z][\w'-]*", s)
    if not 1 <= len(words) <= 8:
        return False
    if s.isupper() and len(s) <= 50:
        return True
    upper_count = sum(
        1 for w in words
        if w.lower() not in _LOWER_FILLERS and w[:1].isupper()
    )
    non_filler = sum(1 for w in words if w.lower() not in _LOWER_FILLERS)
    if non_filler == 0:
        return False
    return upper_count >= max(1, non_filler - 0)         # all non-fillers capped


# Inline tags emitted by _classify_line; keep them stable for downstream RAG.
_QUESTION_RE = re.compile(r'^\s*\d+[.)]\s+.*\?\s*$')
_FIGURE_CAPTION_RE = re.compile(r'^\s*(Figure|Fig\.?|FIGURE)\s+\d', re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r'^\s*(Table|TABLE)\s+\d', re.IGNORECASE)
_MAP_TEXTBOX_RE = re.compile(
    r'^\s*(Map|Sidebar|Skill\s*Builder|Reading\s+Check|Reading\s+Focus|'
    r'Linking\s+Past\s+&?\s*Present|Geography\s+Skill|Why\s+It\s+Matters)\b',
    re.IGNORECASE,
)
# Lines already tagged by an earlier pass -- don't re-tag.
_TAGGED_RE = re.compile(
    r'^\[(Review Question|Caption|Map/Textbox|Figure Description|Page Visual Description)\]'
)


def _classify_line(line: str) -> str:
    """Return ``line`` possibly prefixed with a category tag."""
    s = line.strip()
    if not s or _TAGGED_RE.match(s):
        return line
    if _QUESTION_RE.match(s):
        return f'[Review Question] {s}'
    if _FIGURE_CAPTION_RE.match(s) or _TABLE_CAPTION_RE.match(s):
        return f'[Caption] {s}'
    if _MAP_TEXTBOX_RE.match(s):
        return f'[Map/Textbox] {s}'
    return line


def _annotate_page(
    text: str,
    current_section: str | None,
    running_headers: set[str],
) -> tuple[str, str | None]:
    """Strip running headers from the page text, detect a heading, and
    tag review-question / caption / map lines.

    Returns ``(annotated_text, updated_section)``. The section is updated
    with the *first* in-page heading that isn't a running header, so the
    JSONL section column tracks the most recent sub-heading the reader saw.
    """
    section = current_section
    page_heading: str | None = None

    out: list[str] = []
    for raw in text.split('\n'):
        s = raw.strip()
        if not s:
            out.append('')
            continue
        if s in running_headers:
            continue                                     # silent drop
        # Lock in the first non-header heading we see on this page.
        if page_heading is None and _looks_like_heading(s):
            page_heading = s
        out.append(_classify_line(raw))

    if page_heading:
        section = page_heading

    return '\n'.join(out), section


# ═══════════════════════════════════════════════════════════════════════
# Per-page extraction (digital + quality-gated OCR fallback)
# ═══════════════════════════════════════════════════════════════════════

def _process_page(
    page: fitz.Page,
    doc: fitz.Document,
    *,
    ocr_lang: str,
    dpi: int,
    figure_budget: int,
    describe_pages: bool,
    force_page_analysis: bool,
    vlm_url: str | None = None,
    vlm_model: str | None = None,
    vlm_timeout: int | None = None,
    vlm_diagnostics: dict[str, Any] | None = None,
) -> tuple[str, str, bool, int, int, bool, int, int]:
    """Extract a single page and decide between digital text vs OCR.

    Returns
    ``(text, visual_text, has_visual_description, used_digital, ran_ocr,
    ocr_replaced_bad, embedded_figs_described, page_described)``.
    """
    digital_raw = _read_page_blocks(page)
    digital = _clean_text(digital_raw)

    chosen = digital
    used_digital = 1 if digital else 0
    ran_ocr = 0
    ocr_replaced_bad = False

    if is_bad_extraction(digital):
        ocr_raw = _ocr_page(page, lang=ocr_lang, dpi=dpi)
        if ocr_raw:
            ran_ocr = 1
            ocr_clean = _clean_text(ocr_raw)
            # Switch to OCR if it's not also bad. We *don't* gate purely on
            # length -- a short clean OCR result beats a long broken one.
            if not is_bad_extraction(ocr_clean) and len(ocr_clean) >= 20:
                chosen = ocr_clean
                used_digital = 0
                ocr_replaced_bad = bool(digital.strip())

    cleaned = _post_ocr_cleanup(chosen)
    compact_text = re.sub(r'\s+', ' ', cleaned or '').strip()
    text_rich_page = len(compact_text) >= VISUAL_TEXT_RICH_THRESHOLD

    visual_segments: list[str] = []
    embedded_described = 0
    page_described = 0
    visual_heavy = looks_visual_heavy(page, cleaned)
    unlimited_budget = figure_budget < 0
    used_page_visual = False

    if describe_pages and (force_page_analysis or visual_heavy):
        try:
            desc = _describe_page_with_retries(
                page,
                model=vlm_model,
                base_url=vlm_url,
                timeout=vlm_timeout,
                diagnostics=vlm_diagnostics,
            )
        except Exception as exc:
            print(
                f' [!] Page render warning on page {page.number + 1}: '
                f'{type(exc).__name__}: {exc}',
                file=sys.stderr,
            )
            desc = ''
        if desc:
            visual_segments.append(f'[Page Visual Description] {desc}')
            page_described = 1
            used_page_visual = True

    if (figure_budget > 0 or unlimited_budget) and text_rich_page:
        if TEXT_RICH_MAX_FIGURES < 0:
            remaining_budget = figure_budget
        else:
            remaining_budget = min(figure_budget, max(0, TEXT_RICH_MAX_FIGURES))
        if remaining_budget > 0:
            try:
                for img_idx, img_bytes in enumerate(
                    _extract_page_images(page, doc, max_side=TEXT_RICH_IMAGE_SIDE),
                    start=1,
                ):
                    if not unlimited_budget and embedded_described >= remaining_budget:
                        break
                    desc = describe_image(
                        img_bytes,
                        prompt=_visual_prompt('figure'),
                        max_tokens=TEXT_RICH_FIGURE_MAX_TOKENS,
                        context=f'page {page.number + 1} image {img_idx}',
                        model=vlm_model,
                        base_url=vlm_url,
                        timeout=vlm_timeout,
                        diagnostics=vlm_diagnostics,
                    )
                    if desc:
                        visual_segments.append(f'[Figure Description] {desc}')
                        embedded_described += 1
            except Exception:
                pass

    elif figure_budget > 0 or unlimited_budget:
        remaining_budget = figure_budget if unlimited_budget else max(0, figure_budget - page_described)
        if (unlimited_budget or remaining_budget > 0) and not used_page_visual:
            embedded_attempts = 0
            try:
                for img_idx, img_bytes in enumerate(_extract_page_images(page, doc), start=1):
                    if not unlimited_budget and embedded_attempts >= remaining_budget:
                        break
                    embedded_attempts += 1
                    desc = describe_image(
                        img_bytes,
                        prompt=_visual_prompt('figure'),
                        max_tokens=VLM_FIGURE_MAX_TOKENS,
                        context=f'page {page.number + 1} image {img_idx}',
                        model=vlm_model,
                        base_url=vlm_url,
                        timeout=vlm_timeout,
                        diagnostics=vlm_diagnostics,
                    )
                    if desc:
                        visual_segments.append(f'[Figure Description] {desc}')
                        embedded_described += 1
            except Exception:
                pass

    visual_text = '\n\n'.join(visual_segments).strip()
    if visual_text:
        cleaned = f'{cleaned}\n\n{visual_text}'.strip() if cleaned else visual_text

    return (
        cleaned,
        visual_text,
        bool(visual_text),
        used_digital,
        ran_ocr,
        ocr_replaced_bad,
        embedded_described,
        page_described,
    )


# ═══════════════════════════════════════════════════════════════════════
# Top-level pipeline
# ═══════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: str,
    out_path: str | None = None,
    *,
    jsonl_path: str | None = None,
    dpi: int | None = None,
    ocr_lang: str | None = None,
    max_figures: int | None = None,
    describe_pages: bool | None = None,
    force_page_analysis: bool | None = None,
    vlm_timeout: int | None = None,
    vlm_model: str | None = None,
    vlm_url: str | None = None,
) -> dict[str, Any]:
    pdf_path = os.path.abspath(pdf_path)
    if out_path is None:
        out_path = f'{pdf_path}_extracted.txt'
    if jsonl_path is None:
        jsonl_path = f'{pdf_path}_extracted.jsonl'

    eff_dpi = dpi or PDF_DPI
    eff_ocr_lang = ocr_lang or OCR_LANG
    eff_max_figures = MAX_FIGURES if max_figures is None else max_figures
    eff_describe_pages = DESCRIBE_PAGES if describe_pages is None else describe_pages
    eff_force_page_analysis = FORCE_PAGE_ANALYSIS if force_page_analysis is None else force_page_analysis
    eff_vlm_timeout = REQUEST_TIMEOUT if vlm_timeout is None else int(vlm_timeout)
    eff_vlm_model = vlm_model or VISION_MODEL
    eff_vlm_url = (vlm_url or LLAMA_BASE_URL).rstrip('/')
    vlm_enabled = eff_max_figures != 0
    page_visual_enabled = bool(vlm_enabled and eff_describe_pages)

    print('=' * 60)
    print('  PDF -> Text + JSONL  (pymupdf + Tesseract OCR)')
    print('=' * 60)
    print(f'  Input    : {pdf_path}')
    print(f'  DPI      : {eff_dpi}')
    print(f'  OCR lang : {eff_ocr_lang}')
    print(f'  Tessdata : {TESSDATA_DIR}')
    if vlm_enabled:
        print(f'  VLM      : enabled ({eff_vlm_model} @ {eff_vlm_url})')
        print(f'  Timeout  : {eff_vlm_timeout}s')
        print(f'  Page VLM : {"enabled" if page_visual_enabled else "disabled"}')
        print(f'  Force Pg : {"enabled" if eff_force_page_analysis else "disabled"}')
    else:
        print('  VLM      : disabled')
    print('=' * 60)

    if vlm_enabled:
        try:
            _ensure_vlm_ready(
                model=eff_vlm_model,
                base_url=eff_vlm_url,
                timeout=max(30, min(eff_vlm_timeout, 120)),
            )
        except RuntimeError as exc:
            print(f'Cannot start or reach the VLM server: {exc}', file=sys.stderr)
            sys.exit(3)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        print(f'Cannot open PDF: {exc}', file=sys.stderr)
        sys.exit(2)

    total_pages = len(doc)
    print(f'  Processing {total_pages} page(s) ...')

    # Pass 1: extract + initial cleanup per page (no cross-page knowledge yet).
    per_page_text: list[str] = []
    per_page_visual_text: list[str] = []
    per_page_has_visual: list[bool] = []
    digital_used = 0
    ocr_used = 0
    ocr_replaced = 0
    described = 0
    embedded_described = 0
    page_visuals_described = 0
    vlm_diagnostics: dict[str, Any] = {'vlm_failures': 0}
    for idx in tqdm(range(total_pages), desc='Pages', unit='page'):
        page = doc[idx]
        (
            text,
            visual_text,
            has_visual_description,
            used_digital,
            ran_ocr,
            replaced,
            n_embedded_described,
            n_page_described,
        ) = _process_page(
            page, doc,
            ocr_lang=eff_ocr_lang,
            dpi=eff_dpi,
            figure_budget=eff_max_figures - described,
            describe_pages=page_visual_enabled,
            force_page_analysis=eff_force_page_analysis,
            vlm_url=eff_vlm_url,
            vlm_model=eff_vlm_model,
            vlm_timeout=eff_vlm_timeout,
            vlm_diagnostics=vlm_diagnostics,
        )
        digital_used += used_digital
        ocr_used += ran_ocr
        ocr_replaced += int(replaced)
        embedded_described += n_embedded_described
        page_visuals_described += n_page_described
        described += n_embedded_described + n_page_described
        per_page_text.append(text)
        per_page_visual_text.append(visual_text)
        per_page_has_visual.append(has_visual_description)
    doc.close()

    # Pass 2: find lines that recur on many pages -- those are running headers.
    running_headers = _detect_running_headers(per_page_text)

    # Pass 3: strip running headers + textbook noise, annotate, and number pages.
    txt_chunks: list[str] = []
    jsonl_rows: list[dict[str, Any]] = []
    pages_kept = 0
    section_state: str | None = None
    source_basename = os.path.basename(pdf_path)

    for idx, page_text in enumerate(per_page_text):
        cleaned = _strip_textbook_noise(page_text, running_headers)
        cleaned = _MULTI_BLANK_RE.sub('\n\n', cleaned).strip()
        annotated, section_state = _annotate_page(cleaned, section_state, running_headers)
        annotated = annotated.strip()
        if not annotated:
            continue
        visual_text = _MULTI_BLANK_RE.sub('\n\n', per_page_visual_text[idx]).strip()
        pages_kept += 1
        txt_chunks.append(f'--- Page {idx + 1} ---\n{annotated}')
        jsonl_rows.append({
            'source': source_basename,
            'page': idx + 1,
            'section': section_state,
            'text': annotated,
            'visual_text': visual_text,
            'has_visual_description': per_page_has_visual[idx],
        })

    body = '\n\n'.join(txt_chunks).strip() + '\n'
    header = f'Document: {source_basename}'
    full_txt = f'{header}\n{"=" * 60}\n\n{body}'

    Path(out_path).write_text(full_txt, encoding='utf-8')
    with open(jsonl_path, 'w', encoding='utf-8') as fh:
        for row in jsonl_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')

    # ── Report ─────────────────────────────────────────────────────────
    print('=' * 60)
    print(f'  Total pages              : {total_pages}')
    print(f'  Pages kept (with text)   : {pages_kept}')
    print(f'  Digital pages used       : {digital_used}')
    print(f'  OCR pages used           : {ocr_used}')
    print(f'  OCR replaced bad digital : {ocr_replaced}')
    print(f'  Visuals described total  : {described}')
    print(f'  Embedded figures desc.   : {embedded_described}')
    print(f'  Full pages described     : {page_visuals_described}')
    print(f'  VLM failures             : {vlm_diagnostics["vlm_failures"]}')
    print(f'  VLM enabled              : {vlm_enabled}')
    print(f'  Page visual used         : {bool(page_visuals_described)}')
    print(f'  Running headers stripped : {len(running_headers)}')
    print(f'  Output chars             : {len(full_txt):,}')
    print(f'  Output TXT               : {out_path}')
    print(f'  Output JSONL             : {jsonl_path}')
    print('=' * 60)

    return {
        'total_pages': total_pages,
        'pages_kept': pages_kept,
        'digital_pages': digital_used,
        'ocr_pages': ocr_used,
        'ocr_replaced_bad': ocr_replaced,
        'figures_described': described,
        'embedded_figures_described': embedded_described,
        'page_visuals_described': page_visuals_described,
        'vlm_failures': int(vlm_diagnostics['vlm_failures']),
        'vlm_enabled': vlm_enabled,
        'page_visual_used': bool(page_visuals_described),
        'running_headers': sorted(running_headers),
        'out_txt': out_path,
        'out_jsonl': jsonl_path,
        'output_chars': len(full_txt),
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Convert a PDF to RAG-ready text + JSONL.',
    )
    p.add_argument('pdf', nargs='?', help='Path to the PDF (prompted if omitted).')
    p.add_argument('--out', help='TXT output path (default: <pdf>_extracted.txt)')
    p.add_argument('--jsonl', help='JSONL output path (default: <pdf>_extracted.jsonl)')
    p.add_argument('--dpi', type=int, help=f'Render DPI for OCR (default: {PDF_DPI})')
    p.add_argument('--ocr-lang', dest='ocr_lang',
                   help=f'Tesseract language code; combine with "+" '
                        f'(default: {OCR_LANG})')
    p.add_argument('--max-figures', dest='max_figures', type=int,
                   help=f'Max figures to describe via VLM '
                        f'(default: {MAX_FIGURES}; 0 disables)')
    p.add_argument(
        '--describe-pages',
        dest='describe_pages',
        action='store_true',
        default=None,
        help='Render full-page screenshots for visual-heavy pages '
             '(also requires --max-figures > 0).',
    )
    p.add_argument(
        '--force-page-analysis',
        dest='force_page_analysis',
        action='store_true',
        default=None,
        help='Always run a full-page VLM analysis for every page when page descriptions are enabled.',
    )
    p.add_argument(
        '--vlm-timeout',
        dest='vlm_timeout',
        type=int,
        help=f'VLM request timeout in seconds (default: {REQUEST_TIMEOUT})',
    )
    p.add_argument(
        '--vlm-model',
        dest='vlm_model',
        help=f'Override the local VLM model for this run (default: {VISION_MODEL})',
    )
    p.add_argument(
        '--vlm-url',
        dest='vlm_url',
        help=f'Override the local OpenAI-compatible VLM URL '
             f'(default: {LLAMA_BASE_URL})',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    pdf_input = args.pdf
    if not pdf_input:
        print('Drag and drop your PDF here, or type the path:')
        pdf_input = input('>> ').strip()
    pdf_input = pdf_input.strip().strip('"').strip("'")
    if not pdf_input:
        print('No PDF given.', file=sys.stderr)
        return 1
    if not os.path.exists(pdf_input):
        print(f'File not found: {pdf_input}', file=sys.stderr)
        return 1
    if not pdf_input.lower().endswith('.pdf'):
        print('Please provide a .pdf file.', file=sys.stderr)
        return 1

    process_pdf(
        pdf_input,
        out_path=args.out,
        jsonl_path=args.jsonl,
        dpi=args.dpi,
        ocr_lang=args.ocr_lang,
        max_figures=args.max_figures,
        describe_pages=args.describe_pages,
        force_page_analysis=args.force_page_analysis,
        vlm_timeout=args.vlm_timeout,
        vlm_model=args.vlm_model,
        vlm_url=args.vlm_url,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
