import { getStoredAuthToken } from '../auth';
import { apiRequest } from './client';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export function listDocuments(sessionId) {
  return apiRequest(`/sessions/${sessionId}/documents`);
}

export function uploadDocuments(sessionId, files, options = {}) {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `${API_BASE_URL}/sessions/${sessionId}/documents/upload`);

    const token = getStoredAuthToken();
    if (token) {
      request.setRequestHeader('Authorization', `Bearer ${token}`);
    }

    request.upload.addEventListener('progress', (event) => {
      if (!event.lengthComputable) return;
      options.onUploadProgress?.({
        loaded: event.loaded,
        total: event.total,
        progress: event.total > 0 ? (event.loaded / event.total) * 100 : 0,
      });
    });

    request.upload.addEventListener('load', () => {
      options.onProcessingStart?.();
    });

    request.addEventListener('error', () => {
      reject(new Error('Unable to reach the backend API. Start the full app with `npm run dev` or make sure the backend is running.'));
    });

    request.addEventListener('abort', () => {
      reject(new Error('Upload was cancelled.'));
    });

    request.addEventListener('load', () => {
      const responseText = request.responseText || '';
      let payload = null;
      if (responseText) {
        try {
          payload = JSON.parse(responseText);
        } catch (_error) {
          payload = null;
        }
      }

      if (request.status >= 200 && request.status < 300) {
        resolve(payload);
        return;
      }

      reject(new Error(payload?.detail || request.statusText || 'Request failed.'));
    });

    request.send(form);
  });
}

export function activateCorpus(sessionId, documentIds) {
  return apiRequest(`/sessions/${sessionId}/corpus/activate`, {
    method: 'POST',
    body: JSON.stringify({ document_ids: documentIds }),
  });
}

export function getActiveCorpus(sessionId) {
  return apiRequest(`/sessions/${sessionId}/corpus/active`);
}

export function deleteDocument(sessionId, documentId) {
  return apiRequest(`/sessions/${sessionId}/documents/${documentId}`, {
    method: 'DELETE',
  });
}

export function askGroundedQuestion(sessionId, question) {
  return apiRequest(`/sessions/${sessionId}/learning/ask`, {
    method: 'POST',
    body: JSON.stringify({ question }),
  });
}

export function generateQuestion(sessionId) {
  return apiRequest(`/sessions/${sessionId}/learning/questions`, {
    method: 'POST',
  });
}

export function listGeneratedQuestions(sessionId) {
  return apiRequest(`/sessions/${sessionId}/learning/questions`);
}

export function gradeQuestion(sessionId, questionId, userAnswer) {
  return apiRequest(`/sessions/${sessionId}/learning/questions/${questionId}/grade`, {
    method: 'POST',
    body: JSON.stringify({ user_answer: userAnswer }),
  });
}

export function listAttempts(sessionId) {
  return apiRequest(`/sessions/${sessionId}/learning/attempts`);
}
