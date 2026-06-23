import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';

import { AuthProvider, useAuth } from './auth';
import AppShell from './components/layout/AppShell';
import ChatPage from './pages/ChatPage';
import AuthPage from './pages/AuthPage';
import LearningPage from './pages/LearningPage';
import ProfilePage from './pages/ProfilePage';
import SettingsPage from './pages/SettingsPage';
import { PreferencesProvider, usePreferences } from './preferences';
import { SessionProvider, useSession } from './session';

function AppRoutes() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { session, sessions, error, createNewSession, switchSession, deleteExistingSession } = useSession();
  const { t } = usePreferences();

  const assistantActive = location.pathname.startsWith('/assistant');

  const handleCreateSession = async () => {
    const created = await createNewSession();
    navigate('/assistant');
    return created;
  };

  const handleSelectSession = async (nextSessionId) => {
    await switchSession(nextSessionId);
    navigate('/assistant');
  };

  const handleOpenProfile = () => {
    navigate('/profile');
  };

  const handleOpenSettings = () => {
    navigate('/settings');
  };

  return (
    <AppShell
      sessionId={session?.session_id}
      sessions={sessions}
      user={user}
      onOpenProfile={handleOpenProfile}
      onOpenSettings={handleOpenSettings}
      onLogout={logout}
      onCreateSession={handleCreateSession}
      onSelectSession={handleSelectSession}
      onDeleteSession={deleteExistingSession}
      showSessionRail={assistantActive}
      nav={
        <>
          <NavLink to="/" end className={({ isActive }) => `nav-link nav-link-primary ${isActive ? 'active' : ''}`.trim()}>
            <span className="nav-link-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" className="nav-link-svg">
                <path d="M5 6.5h14M5 12h14M5 17.5h14" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
              </svg>
            </span>
            <span className="nav-link-copy-group">
              <span className="nav-link-title">{t('ragStudio')}</span>
              <span className="nav-link-copy">{t('ragStudioDescription')}</span>
            </span>
          </NavLink>
          <NavLink to="/assistant" className={({ isActive }) => `nav-link nav-link-primary ${isActive ? 'active' : ''}`.trim()}>
            <span className="nav-link-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" className="nav-link-svg">
                <path d="M12 3.5c4.97 0 9 3.246 9 7.25S16.97 18 12 18a10.2 10.2 0 0 1-3.084-.47L4 19.5l1.516-3.41A6.65 6.65 0 0 1 3 10.75C3 6.746 7.03 3.5 12 3.5Z" fill="none" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.5" />
              </svg>
            </span>
            <span className="nav-link-copy-group">
              <span className="nav-link-title">{t('assistant')}</span>
              <span className="nav-link-copy">{t('assistantDescription')}</span>
            </span>
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => `nav-link nav-link-primary ${isActive ? 'active' : ''}`.trim()}>
            <span className="nav-link-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" className="nav-link-svg">
                <path d="M12 8.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7Zm8.25 3.5a7.85 7.85 0 0 0-.08-1.1l1.74-1.35-1.75-3.03-2.05.82a8.15 8.15 0 0 0-1.9-1.1L15.9 4h-3.5l-.31 2.24a8.15 8.15 0 0 0-1.9 1.1l-2.05-.82-1.75 3.03 1.74 1.35a7.85 7.85 0 0 0 0 2.2l-1.74 1.35 1.75 3.03 2.05-.82a8.15 8.15 0 0 0 1.9 1.1L12.4 20h3.5l.31-2.24a8.15 8.15 0 0 0 1.9-1.1l2.05.82 1.75-3.03-1.74-1.35c.05-.36.08-.73.08-1.1Z" fill="none" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.4" />
              </svg>
            </span>
            <span className="nav-link-copy-group">
              <span className="nav-link-title">{t('settings')}</span>
              <span className="nav-link-copy">{t('settingsSubtitle')}</span>
            </span>
          </NavLink>
        </>
      }
      sessionError={error}
    >
      <Routes>
        <Route path="/" element={<LearningPage />} />
        <Route path="/assistant" element={<ChatPage />} />
        <Route path="/profile" element={<ProfilePage user={user} onLogout={logout} />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/learning" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}

function AuthenticatedApp() {
  const { user, loading } = useAuth();
  const { t } = usePreferences();

  if (loading) {
    return (
      <div className="auth-shell auth-shell-loading">
        <div className="auth-loading-card">
          <div className="auth-brand-mark">EA</div>
          <p className="eyebrow">Edge AI</p>
          <h2>{t('restoringWorkspace')}</h2>
        </div>
      </div>
    );
  }

  if (!user) {
    return <AuthPage />;
  }

  return (
    <SessionProvider storageNamespace={user.id}>
      <AppRoutes />
    </SessionProvider>
  );
}

export default function App() {
  return (
    <PreferencesProvider>
      <AuthProvider>
        <AuthenticatedApp />
      </AuthProvider>
    </PreferencesProvider>
  );
}
