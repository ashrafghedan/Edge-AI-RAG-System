import { createContext, useContext, useEffect, useMemo, useState } from 'react';

import { createSession, deleteSession, getSession, listSessions } from './api/sessions';

const SessionContext = createContext(null);

function getStorageKey(storageNamespace) {
  return `edge-rag-session-id:${storageNamespace || 'default'}`;
}

export function SessionProvider({ children, storageNamespace }) {
  const storageKey = getStorageKey(storageNamespace);
  const [sessionId, setSessionId] = useState(() => window.localStorage.getItem(storageKey));
  const [session, setSession] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setSessionId(window.localStorage.getItem(storageKey));
    setSession(null);
    setSessions([]);
    setLoading(true);
    setError('');
  }, [storageKey]);

  async function refreshSessionListSafe() {
    try {
      const listing = await listSessions();
      setSessions(Array.isArray(listing) ? listing : []);
      return Array.isArray(listing) ? listing : [];
    } catch (_error) {
      return [];
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function ensureSession() {
      setLoading(true);
      setError('');
      try {
        if (!sessionId) {
          const created = await createSession();
          if (cancelled) return;
          window.localStorage.setItem(storageKey, created.session_id);
          setSessionId(created.session_id);
          setSession(created);
          await refreshSessionListSafe();
          return;
        }

        try {
          const existing = await getSession(sessionId);
          if (!cancelled) {
            setSession(existing);
          }
          await refreshSessionListSafe();
          return;
        } catch (_error) {
          const created = await createSession();
          if (cancelled) return;
          window.localStorage.setItem(storageKey, created.session_id);
          setSessionId(created.session_id);
          setSession(created);
          await refreshSessionListSafe();
        }
      } catch (_error) {
        if (!cancelled) {
          setSession(null);
          setError('Unable to initialize a session. Start the backend with `npm run dev` and refresh the page.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    ensureSession();
    return () => {
      cancelled = true;
    };
  }, [sessionId, storageKey]);

  const value = useMemo(
    () => ({
      sessionId,
      session,
      sessions,
      loading,
      error,
      refreshSession: async () => {
        if (!sessionId) return null;
        const refreshed = await getSession(sessionId);
        setSession(refreshed);
        await refreshSessionListSafe();
        setError('');
        return refreshed;
      },
      createNewSession: async () => {
        const created = await createSession();
        window.localStorage.setItem(storageKey, created.session_id);
        setSessionId(created.session_id);
        setSession(created);
        await refreshSessionListSafe();
        setError('');
        return created;
      },
      switchSession: async (nextSessionId) => {
        if (!nextSessionId || nextSessionId === sessionId) return;
        window.localStorage.setItem(storageKey, nextSessionId);
        setSessionId(nextSessionId);
        setLoading(true);
        setError('');
      },
      deleteExistingSession: async (targetSessionId) => {
        if (!targetSessionId) return;
        await deleteSession(targetSessionId);
        const listing = await refreshSessionListSafe();
        const remainingSessions = listing.filter((item) => item.session_id !== targetSessionId);
        setSessions(remainingSessions);
        setError('');

        if (targetSessionId !== sessionId) {
          return;
        }

        const nextSession = remainingSessions[0];
        if (nextSession) {
          window.localStorage.setItem(storageKey, nextSession.session_id);
          setSessionId(nextSession.session_id);
          setLoading(true);
          return;
        }

        window.localStorage.removeItem(storageKey);
        setSession(null);
        setSessionId(null);
        setLoading(true);
      },
    }),
    [sessionId, session, sessions, loading, error, storageKey],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const value = useContext(SessionContext);
  if (!value) {
    throw new Error('useSession must be used inside SessionProvider.');
  }
  return value;
}
