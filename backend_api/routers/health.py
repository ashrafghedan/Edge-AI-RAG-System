from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from edge_rag.app import EdgeRagApp


router = APIRouter(tags=['health'])


@router.get('/health')
def health_check() -> dict[str, object]:
    app = EdgeRagApp()
    llama_ok, llama_message = app.system_status()
    loaded_models, _ = app.clients.available_models()
    config = app.config.models
    mmproj_path = config.mmproj_path or ''
    vision_ready = bool(config.vision_enabled and mmproj_path and Path(mmproj_path).is_file())
    return {
        'status': 'ok',
        'llama_cpp_available': llama_ok,
        'llama_cpp_message': llama_message,
        'llama_cpp_base_url': config.ollama_base_url,
        'model': config.answer_model,
        'loaded_models': loaded_models,
        'streaming_enabled': bool(config.stream_enabled),
        'vision_enabled': vision_ready,
        'vision_message': (
            'Multimodal projector loaded.'
            if vision_ready
            else 'Image input disabled (set LLAMA_CPP_VISION_ENABLED=1 and provide a valid LLAMA_CPP_MMPROJ_PATH).'
        ),
        'mmproj_path': mmproj_path if vision_ready else '',
        'reasoning_mode': config.reasoning_mode,
        # Legacy aliases preserved for any existing client.
        'ollama_available': llama_ok,
        'ollama_message': llama_message,
    }
