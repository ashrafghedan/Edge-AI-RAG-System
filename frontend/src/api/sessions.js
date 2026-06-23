import { apiRequest } from './client';

export function listSessions() {
  return apiRequest('/sessions');
}

export function createSession() {
  return apiRequest('/sessions', { method: 'POST' });
}

export function getSession(sessionId) {
  return apiRequest(`/sessions/${sessionId}`);
}

export function deleteSession(sessionId) {
  return apiRequest(`/sessions/${sessionId}`, { method: 'DELETE' });
}
