import { createContext, useContext, useEffect, useMemo, useState } from 'react';

import { getCurrentUser, loginRequest, logoutRequest, registerRequest } from './api/auth';

const SESSION_KEY = 'edge-rag-auth-session';
const AuthContext = createContext(null);

function loadAuthSession() {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed?.token || !parsed?.user?.id) {
      return null;
    }
    return parsed;
  } catch (_error) {
    return null;
  }
}

function saveAuthSession(session) {
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

function clearAuthSession() {
  window.localStorage.removeItem(SESSION_KEY);
}

function normalizeUser(user) {
  return {
    id: user.user_id,
    email: user.email,
    name: user.display_name,
    initials: user.initials,
    createdAt: user.created_at,
  };
}

export function getStoredAuthToken() {
  return loadAuthSession()?.token || '';
}

export function AuthProvider({ children }) {
  const [authSession, setAuthSession] = useState(() => loadAuthSession());
  const [loading, setLoading] = useState(() => Boolean(loadAuthSession()?.token));

  useEffect(() => {
    let cancelled = false;
    const current = loadAuthSession();
    if (!current?.token) {
      setLoading(false);
      return () => {
        cancelled = true;
      };
    }

    async function restore() {
      try {
        const user = await getCurrentUser();
        if (cancelled) return;
        const nextSession = { token: current.token, user: normalizeUser(user) };
        saveAuthSession(nextSession);
        setAuthSession(nextSession);
      } catch (_error) {
        if (!cancelled) {
          clearAuthSession();
          setAuthSession(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo(
    () => ({
      user: authSession?.user || null,
      token: authSession?.token || '',
      authenticated: Boolean(authSession?.token && authSession?.user),
      loading,
      login: async ({ email, password }) => {
        const payload = await loginRequest(email, password);
        const nextSession = {
          token: payload.access_token,
          user: normalizeUser(payload.user),
        };
        saveAuthSession(nextSession);
        setAuthSession(nextSession);
        setLoading(false);
        return nextSession;
      },
      register: async ({ name, email, password }) => {
        const payload = await registerRequest(name, email, password);
        const nextSession = {
          token: payload.access_token,
          user: normalizeUser(payload.user),
        };
        saveAuthSession(nextSession);
        setAuthSession(nextSession);
        setLoading(false);
        return nextSession;
      },
      logout: async () => {
        try {
          await logoutRequest();
        } catch (_error) {
        } finally {
          clearAuthSession();
          setAuthSession(null);
          setLoading(false);
        }
      },
    }),
    [authSession, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error('useAuth must be used inside AuthProvider.');
  }
  return value;
}
