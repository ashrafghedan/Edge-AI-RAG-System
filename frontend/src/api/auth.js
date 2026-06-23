import { apiRequest } from './client';

export function loginRequest(email, password) {
  return apiRequest('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
}

export function registerRequest(displayName, email, password) {
  return apiRequest('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ display_name: displayName, email, password }),
  });
}

export function getCurrentUser() {
  return apiRequest('/auth/me');
}

export function logoutRequest() {
  return apiRequest('/auth/logout', { method: 'POST' });
}
