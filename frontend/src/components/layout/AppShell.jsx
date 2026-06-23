import { useEffect, useRef, useState } from 'react';

import { usePreferences } from '../../preferences';

export default function AppShell({
  sessionError,
  sessionId,
  sessions,
  user,
  onOpenProfile,
  onOpenSettings,
  onLogout,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
  showSessionRail,
  nav,
  children,
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const { t } = usePreferences();

  useEffect(() => {
    function handlePointerDown(event) {
      if (!menuRef.current?.contains(event.target)) {
        setMenuOpen(false);
      }
    }

    function handleEscape(event) {
      if (event.key === 'Escape') {
        setMenuOpen(false);
      }
    }

    window.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('keydown', handleEscape);
    return () => {
      window.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('keydown', handleEscape);
    };
  }, []);

  const userInitial = (user?.name || user?.email || t('initial')).trim().charAt(0).toUpperCase();

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="sidebar-top">
          <div className="sidebar-app-label">
            <span className="status-label">Edge AI</span>
          </div>

          <div className="user-menu" ref={menuRef}>
            <button
              type="button"
              className={`user-menu-trigger ${menuOpen ? 'user-menu-trigger-open' : ''}`}
              onClick={() => setMenuOpen((current) => !current)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
            >
              <span className="user-avatar" aria-hidden="true">
                {userInitial}
              </span>
              <span className="user-meta">
                <strong>{user?.name || t('user')}</strong>
                <span>{user?.email || t('signedIn')}</span>
              </span>
              <span className="user-menu-chevron" aria-hidden="true">
                <svg viewBox="0 0 24 24" className="nav-link-svg">
                  <path d="m7 10 5 5 5-5" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
                </svg>
              </span>
            </button>

            {menuOpen ? (
              <div className="user-menu-dropdown" role="menu">
                <div className="user-menu-profile">
                  <span className="user-avatar user-avatar-large" aria-hidden="true">
                    {userInitial}
                  </span>
                  <div className="user-menu-profile-copy">
                    <strong>{user?.name || t('user')}</strong>
                    <span>{user?.email || t('signedIn')}</span>
                    <small>{t('profile')}</small>
                  </div>
                </div>
                <button
                  type="button"
                  className="user-menu-action"
                  onClick={() => {
                    setMenuOpen(false);
                    onOpenProfile?.();
                  }}
                >
                  {t('viewProfile')}
                </button>
                <button
                  type="button"
                  className="user-menu-action"
                  onClick={() => {
                    setMenuOpen(false);
                    onOpenSettings?.();
                  }}
                >
                  {t('settings')}
                </button>
                <button
                  type="button"
                  className="user-menu-action user-menu-action-danger"
                  onClick={() => {
                    setMenuOpen(false);
                    onLogout?.();
                  }}
                >
                  {t('logout')}
                </button>
              </div>
            ) : null}
          </div>
        </div>

        <nav className="app-nav">{nav}</nav>

        {showSessionRail ? (
          <section className="session-rail">
            <button type="button" className="secondary-button new-chat-button" onClick={() => onCreateSession?.()}>
              {t('newChat')}
            </button>
            <div className="session-list-header">
              <span className="status-label">{t('recentChats')}</span>
            </div>
            <div className="session-list">
              {sessions?.length ? (
                sessions.map((item) => (
                  <div
                    key={item.session_id}
                    className={`session-entry ${item.session_id === sessionId ? 'session-entry-active' : ''}`}
                  >
                    <button
                      type="button"
                      className={`session-item ${item.session_id === sessionId ? 'session-item-active' : ''}`}
                      onClick={() => onSelectSession?.(item.session_id)}
                    >
                      <strong>{item.title}</strong>
                      <span>{item.active_corpus_label || t('generalChat')}</span>
                    </button>
                    <button
                      type="button"
                      className="session-delete"
                      aria-label={`Delete ${item.title}`}
                      onClick={() => {
                        const confirmed = window.confirm(t('confirmDelete', { name: item.title }));
                        if (!confirmed) return;
                        onDeleteSession?.(item.session_id);
                      }}
                    >
                      <svg viewBox="0 0 24 24" className="nav-link-svg" aria-hidden="true">
                        <path d="M6 7h12M9.5 7V5.75A1.75 1.75 0 0 1 11.25 4h1.5A1.75 1.75 0 0 1 14.5 5.75V7M8.5 10v7M12 10v7M15.5 10v7M7.5 7l.75 11.25A1.75 1.75 0 0 0 10 20h4a1.75 1.75 0 0 0 1.75-1.75L16.5 7" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.6" />
                      </svg>
                    </button>
                  </div>
                ))
              ) : (
                <div className="session-empty">{t('noSessions')}</div>
              )}
            </div>
          </section>
        ) : (
          <div className="sidebar-section-placeholder" aria-hidden="true" />
        )}

        {sessionError ? <p className="status-error sidebar-error">{sessionError}</p> : null}
      </aside>
      <main className="app-main">{children}</main>
    </div>
  );
}
