import { getStoredAuthToken } from '../auth';
import { apiRequest } from './client';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export function listChatMessages(sessionId) {
  return apiRequest(`/sessions/${sessionId}/chat/messages`);
}

export function sendChatMessage(sessionId, message, files = [], options = {}) {
  const { signal } = options;

  if (files.length) {
    const formData = new FormData();
    formData.append('message', message);
    files.forEach((file) => formData.append('files', file));
    return sendChatRequest(sessionId, {
      method: 'POST',
      body: formData,
      signal,
    });
  }

  return sendChatRequest(sessionId, {
    method: 'POST',
    body: JSON.stringify({ message }),
    signal,
  });
}

/**
 * Stream a chat reply from `/messages/stream` as Server-Sent Events.
 *
 * Calls `onEvent({type, ...})` for each parsed event from the backend:
 *   - {type: 'analyzing'}
 *   - {type: 'user', message: {...}}
 *   - {type: 'token', text: '...'}
 *   - {type: 'replace', text: '...'}     (final policy rewrite)
 *   - {type: 'meta', tokens, tokens_per_second, elapsed_seconds, ...}
 *   - {type: 'done', message: {...}, stats: {...}}
 *   - {type: 'error', detail: '...'}
 */
export async function streamChatMessage(sessionId, message, files = [], options = {}) {
  const { signal, onEvent } = options;
  const url = `${API_BASE_URL}/sessions/${sessionId}/chat/messages/stream`;

  let body;
  const headers = {
    Accept: 'text/event-stream',
    ...(getStoredAuthToken() ? { Authorization: `Bearer ${getStoredAuthToken()}` } : {}),
  };
  if (files.length) {
    const formData = new FormData();
    formData.append('message', message);
    files.forEach((file) => formData.append('files', file));
    body = formData;
  } else {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify({ message });
  }

  const response = await fetch(url, { method: 'POST', headers, body, signal }).catch((error) => {
    if (error?.name === 'AbortError') {
      throw error;
    }
    throw new Error('Unable to reach the backend API. Start the full app with `npm run dev` or make sure the backend is running.');
  });

  if (!response.ok || !response.body) {
    let detail = 'Streaming request failed.';
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_error) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let separatorIndex;
    // SSE events are separated by a blank line.
    while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      const data = parseSseEventBlock(rawEvent);
      if (data === null) continue;
      try {
        const parsed = JSON.parse(data);
        onEvent?.(parsed);
        if (parsed?.type === 'error') {
          throw new Error(parsed.detail || 'Streaming failed.');
        }
      } catch (parseError) {
        if (parseError instanceof SyntaxError) {
          continue;
        }
        throw parseError;
      }
    }
  }
}

function parseSseEventBlock(block) {
  const lines = block.split('\n');
  const dataLines = [];
  for (const line of lines) {
    if (!line.startsWith('data:')) continue;
    dataLines.push(line.slice(5).replace(/^\s/, ''));
  }
  if (!dataLines.length) return null;
  return dataLines.join('\n');
}

export async function transcribeChatAudio(sessionId, audioFile, { language } = {}) {
  const formData = new FormData();
  formData.append('file', audioFile);
  if (language) {
    formData.append('language', language);
  }

  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/chat/transcribe`, {
    method: 'POST',
    body: formData,
    headers: {
      ...(getStoredAuthToken() ? { Authorization: `Bearer ${getStoredAuthToken()}` } : {}),
    },
  }).catch((error) => {
    if (error?.name === 'AbortError') {
      throw error;
    }
    throw new Error('Unable to reach the backend API. Start the full app with `npm run dev` or make sure the backend is running.');
  });

  if (!response.ok) {
    let detail = 'Request failed.';
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_error) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  return response.json();
}

async function sendChatRequest(sessionId, options) {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/chat/messages`, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(getStoredAuthToken() ? { Authorization: `Bearer ${getStoredAuthToken()}` } : {}),
      ...(options.headers || {}),
    },
  }).catch((error) => {
    if (error?.name === 'AbortError') {
      throw error;
    }
    throw new Error('Unable to reach the backend API. Start the full app with `npm run dev` or make sure the backend is running.');
  });

  if (!response.ok) {
    let detail = 'Request failed.';
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_error) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  return response.json();
}
