import { usePreferences } from '../preferences';

export default function ProfilePage({ user, onLogout }) {
  const { locale, t } = usePreferences();
  const fallbackInitial = (user?.name || user?.email || t('initial')).trim().charAt(0).toUpperCase();

  return (
    <section className="profile-page">
      <header className="page-header page-header-minimal">
        <div>
          <p className="eyebrow">{t('account')}</p>
          <h2>{t('profile')}</h2>
        </div>
      </header>

      <div className="profile-grid">
        <section className="panel surface-card profile-summary-card">
          <div className="profile-summary-header">
            <span className="user-avatar user-avatar-profile" aria-hidden="true">
              {fallbackInitial}
            </span>
            <div className="profile-summary-copy">
              <strong>{user?.name || t('user')}</strong>
              <span>{user?.email || t('signedIn')}</span>
            </div>
          </div>

          <dl className="profile-fields">
            <div className="profile-field">
              <dt>{t('displayName')}</dt>
              <dd>{user?.name || t('notSet')}</dd>
            </div>
            <div className="profile-field">
              <dt>{t('email')}</dt>
              <dd>{user?.email || t('notSet')}</dd>
            </div>
            <div className="profile-field">
              <dt>{t('initials')}</dt>
              <dd>{user?.initials || fallbackInitial}</dd>
            </div>
            <div className="profile-field">
              <dt>{t('accountCreated')}</dt>
              <dd>{user?.createdAt ? new Date(user.createdAt).toLocaleString(locale) : t('unknown')}</dd>
            </div>
          </dl>
        </section>

        <section className="panel surface-card profile-actions-card">
          <div className="panel-header">
            <div>
              <span className="status-label">{t('session')}</span>
              <h3>{t('accountActions')}</h3>
            </div>
          </div>
          <p className="profile-actions-copy">
            {t('profileLogoutHelp')}
          </p>
          <button type="button" className="secondary-button profile-logout-button" onClick={() => onLogout?.()}>
            {t('logout')}
          </button>
        </section>
      </div>
    </section>
  );
}
