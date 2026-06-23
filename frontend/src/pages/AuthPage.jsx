import { useMemo, useState } from 'react';

import { useAuth } from '../auth';
import { usePreferences } from '../preferences';

const INITIAL_MODE = 'login';

export default function AuthPage() {
  const { login, register } = useAuth();
  const { t } = usePreferences();
  const [mode, setMode] = useState(INITIAL_MODE);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);

  const submitLabel = useMemo(() => (mode === 'login' ? t('signIn') : t('createAccount')), [mode, t]);

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setPending(true);

    try {
      if (mode === 'login') {
        await login({ email, password });
      } else {
        if (password !== confirmPassword) {
          throw new Error(t('passwordsDoNotMatch'));
        }
        await register({ name, email, password });
      }
    } catch (submitError) {
      setError(submitError.message || t('unableToContinue'));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-panel auth-panel-hero">
        <div className="auth-brand-mark">EA</div>
        <div className="auth-hero-copy">
          <p className="eyebrow">Edge AI</p>
          <h1>{t('authHero')}</h1>
          <p>{t('authHeroBody')}</p>
        </div>
        <div className="auth-feature-list">
          <div className="auth-feature-card">
            <strong>{t('assistant')}</strong>
            <span>{t('assistantFeature')}</span>
          </div>
          <div className="auth-feature-card">
            <strong>{t('ragStudio')}</strong>
            <span>{t('ragStudioFeature')}</span>
          </div>
        </div>
      </div>

      <div className="auth-panel auth-panel-form">
        <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
          <button
            type="button"
            className={`auth-tab ${mode === 'login' ? 'auth-tab-active' : ''}`}
            onClick={() => {
              setMode('login');
              setError('');
            }}
          >
            {t('signIn')}
          </button>
          <button
            type="button"
            className={`auth-tab ${mode === 'register' ? 'auth-tab-active' : ''}`}
            onClick={() => {
              setMode('register');
              setError('');
            }}
          >
            {t('register')}
          </button>
        </div>

        <div className="auth-form-copy">
          <h2>{mode === 'login' ? t('welcomeBack') : t('createAccountTitle')}</h2>
          <p>{mode === 'login' ? t('signInHelp') : t('createLocalAccountHelp')}</p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === 'register' ? (
            <label className="auth-field">
              <span>{t('name')}</span>
              <input type="text" value={name} onChange={(event) => setName(event.target.value)} placeholder={t('yourFullName')} />
            </label>
          ) : null}

          <label className="auth-field">
            <span>{t('email')}</span>
            <input type="text" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@example.com" autoComplete="username" />
          </label>

          <label className="auth-field">
            <span>{t('password')}</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={mode === 'login' ? t('loginPasswordPlaceholder') : t('useAtLeastEight')}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            />
          </label>

          {mode === 'register' ? (
            <label className="auth-field">
                <span>{t('confirmPassword')}</span>
              <input
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                placeholder={t('repeatPassword')}
                autoComplete="new-password"
              />
            </label>
          ) : null}

          {error ? <p className="error-text auth-error">{error}</p> : null}

          <button type="submit" className="auth-submit" disabled={pending}>
            {pending ? t('pleaseWait') : submitLabel}
          </button>
        </form>
      </div>
    </div>
  );
}
