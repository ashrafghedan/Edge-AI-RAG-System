import { usePreferences } from '../preferences';

export default function SettingsPage() {
  const { language, setLanguage, setTheme, t, theme } = usePreferences();

  return (
    <section className="settings-page">
      <header className="page-header page-header-minimal">
        <div>
          <p className="eyebrow">{t('settings')}</p>
          <h2>{t('settingsSubtitle')}</h2>
        </div>
      </header>

      <div className="settings-grid">
        <section className="panel surface-card settings-card">
          <div className="settings-card-copy">
            <span className="status-label">{t('theme')}</span>
            <h3>{t('appearance')}</h3>
            <p>{t('appearanceHelp')}</p>
          </div>
          <div className="segmented-control" role="group" aria-label={t('theme')}>
            <button
              type="button"
              className={`segmented-option ${theme === 'dark' ? 'segmented-option-active' : ''}`.trim()}
              onClick={() => setTheme('dark')}
            >
              {t('dark')}
            </button>
            <button
              type="button"
              className={`segmented-option ${theme === 'light' ? 'segmented-option-active' : ''}`.trim()}
              onClick={() => setTheme('light')}
            >
              {t('light')}
            </button>
          </div>
        </section>

        <section className="panel surface-card settings-card">
          <div className="settings-card-copy">
            <span className="status-label">{t('language')}</span>
            <h3>{t('appLanguage')}</h3>
            <p>{t('appLanguageHelp')}</p>
          </div>
          <div className="segmented-control" role="group" aria-label={t('language')}>
            <button
              type="button"
              className={`segmented-option ${language === 'en' ? 'segmented-option-active' : ''}`.trim()}
              onClick={() => setLanguage('en')}
            >
              {t('english')}
            </button>
            <button
              type="button"
              className={`segmented-option ${language === 'ar' ? 'segmented-option-active' : ''}`.trim()}
              onClick={() => setLanguage('ar')}
            >
              {t('arabic')}
            </button>
          </div>
        </section>
      </div>
    </section>
  );
}
