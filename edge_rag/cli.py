from __future__ import annotations

import textwrap
from pathlib import Path

from .app import EdgeRagApp
from .types import ActiveCorpus


def _banner(title: str) -> None:
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def _choose_paths(paths: list[Path]) -> list[Path]:
    if len(paths) == 1:
        return paths
    _banner('Choose the text file set to load')
    for index, path in enumerate(paths, start=1):
        print(f'{index}. {path.name}')
    print('a. All files')
    while True:
        choice = input('\nSelect one file number or type a for all: ').strip().lower()
        if choice in {'a', 'all'}:
            return paths
        if choice.isdigit():
            numeric = int(choice)
            if 1 <= numeric <= len(paths):
                return [paths[numeric - 1]]
        print('Invalid selection. Try again.')


def _show_active_corpus(active: ActiveCorpus) -> None:
    _banner('Active text source')
    print(f'Dataset id: {active.dataset_id}')
    if len(active.source_paths) == 1:
        print(f'File: {active.source_paths[0]}')
    else:
        print('Files:')
        for path in active.source_paths:
            print(f'  - {path}')
    print(f'Sources: {", ".join(active.source_names)}')
    print(f'Chunks: {active.chunk_count}')
    print(f'Vector store: {active.vector_directory}')


def _print_wrapped(label: str, value: str) -> None:
    print(textwrap.fill(f'{label}{value}', width=96))


def _activate_from_input(app: EdgeRagApp, raw_path: str) -> ActiveCorpus:
    discovered = app.discover_sources(raw_path)
    selected = _choose_paths(discovered)
    return app.activate_selection(selected)


def _ensure_startup_selection(app: EdgeRagApp) -> ActiveCorpus | None:
    restored = app.restore_last_selection()
    if restored is not None:
        print(f'Restored active text source: {", ".join(restored.source_names)}')
        return restored

    while True:
        _banner('Load text source')
        raw_path = input('Enter a path to a .txt file or folder (or q to quit): ').strip()
        if raw_path.lower() == 'q':
            return None
        if not raw_path:
            print('No path entered.')
            continue
        try:
            active = _activate_from_input(app, raw_path)
        except Exception as exc:
            print(f'Error: {exc}')
            continue
        _show_active_corpus(active)
        return active


def run_cli() -> None:
    app = EdgeRagApp()
    ok, message = app.system_status()
    _banner('Offline Edge RAG Tutor')
    print(message)
    if not ok:
        print('The CLI can still start, but inference calls will fail until llama-server is running with the Gemma model loaded.')
    if _ensure_startup_selection(app) is None:
        print('Exiting.')
        return

    while True:
        _banner('Main menu')
        print('1. Ask a grounded question')
        print('2. Generate a reading question and grade my answer')
        print('3. Show the active TXT file/folder or change it')
        print('q. Quit')
        choice = input('\nChoice: ').strip().lower()

        if choice == '1':
            try:
                app.require_active_corpus()
            except Exception as exc:
                print(f'Error: {exc}')
                continue
            question = input('Enter your question: ').strip()
            if not question:
                print('No question entered.')
                continue
            try:
                result = app.ask_question(question)
            except Exception as exc:
                print(f'Error: {exc}')
                continue
            _banner('Answer')
            _print_wrapped('', result.answer)
            if result.source_names:
                print('\nSources: ' + ', '.join(result.source_names))

        elif choice == '2':
            try:
                app.require_active_corpus()
            except Exception as exc:
                print(f'Error: {exc}')
                continue
            try:
                generated = app.generate_question()
            except Exception as exc:
                message = str(exc)
                if 'non-repeated reading-comprehension question' in message:
                    print('No new unique reading-comprehension question is available for the active TXT source right now.')
                else:
                    print(f'Error: {exc}')
                continue
            _banner('Generated question')
            _print_wrapped('', generated.question)
            user_answer = input('\nYour answer: ').strip()
            if not user_answer:
                print('No answer entered. The generated question was still logged for history tracking.')
                continue
            try:
                grading = app.grade_generated_question(generated.question_id, user_answer)
            except Exception as exc:
                print(f'Error: {exc}')
                continue
            _banner('Grading result')
            print(f'Score: {grading.score}/10')
            _print_wrapped('Feedback: ', grading.feedback)
            _print_wrapped('Model answer: ', grading.model_answer)

        elif choice == '3':
            active = app.state.active_corpus
            if active is None:
                print('No TXT source is currently loaded.')
            else:
                _show_active_corpus(active)
            raw_path = input('\nEnter a new path to change it, or press Enter to keep it: ').strip()
            if not raw_path:
                continue
            try:
                active = _activate_from_input(app, raw_path)
            except Exception as exc:
                print(f'Error: {exc}')
                continue
            _show_active_corpus(active)

        elif choice == 'q':
            print('Exiting.')
            return

        else:
            print('Choose 1, 2, 3, or q.')
