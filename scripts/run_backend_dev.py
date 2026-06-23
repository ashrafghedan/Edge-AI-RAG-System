from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQLITE_FALLBACK_URL = f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').resolve().as_posix()}"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def probe_backend(host: str, port: str, *, timeout: float = 2.0) -> dict[str, object] | None:
    url = f'http://{host}:{port}/api/v1/health'
    request = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def port_is_open(host: str, port: str, *, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def wait_forever() -> int:
    try:
        while True:
            time.sleep(1 << 20)
    except KeyboardInterrupt:
        return 0


def expected_model() -> str:
    return (
        os.getenv('LLAMA_CPP_MODEL')
        or os.getenv('EDGE_RAG_ANSWER_MODEL')
        or 'gemma-4-e2b-q4km'
    ).strip()


def expected_reasoning_mode() -> str:
    return str(os.getenv('LLAMA_CPP_REASONING', 'auto') or 'auto').strip().lower() or 'auto'


def main() -> int:
    load_env_file(PROJECT_ROOT / '.env')
    os.environ.setdefault('DATABASE_URL', SQLITE_FALLBACK_URL)
    backend_port = os.getenv('BACKEND_PORT', '8001')
    backend_host = os.getenv('BACKEND_HOST', '127.0.0.1')

    if health := probe_backend(backend_host, backend_port):
        actual_model = str(health.get('model') or '').strip()
        actual_reasoning = str(health.get('reasoning_mode') or '').strip().lower()
        wanted_model = expected_model()
        wanted_reasoning = expected_reasoning_mode()
        if actual_model == wanted_model and actual_reasoning == wanted_reasoning:
            print(f'[backend] Using existing server at http://{backend_host}:{backend_port}.', flush=True)
            return wait_forever()

        print(
            f'[backend] A backend is already running at http://{backend_host}:{backend_port}, '
            'but its config does not match the current .env.',
            flush=True,
        )
        print(f'[backend] Expected model={wanted_model}, reasoning={wanted_reasoning}.', flush=True)
        print(
            f'[backend] Running  model={actual_model or "(unknown)"}, '
            f'reasoning={actual_reasoning or "(unknown)"}.',
            flush=True,
        )
        print('[backend] Stop the existing backend and run `npm run dev` again.', flush=True)
        return 1

    if port_is_open(backend_host, backend_port):
        print(
            f'[backend] Port {backend_port} is already in use on {backend_host}, '
            'but no healthy Edge RAG backend responded at /api/v1/health.',
            flush=True,
        )
        print('[backend] Free that port or stop the conflicting process, then run `npm run dev` again.', flush=True)
        return 1

    command = [
        sys.executable,
        '-m',
        'uvicorn',
        'backend_api.main:app',
        '--reload',
        '--host',
        backend_host,
        '--port',
        backend_port,
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
