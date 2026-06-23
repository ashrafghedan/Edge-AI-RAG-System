import { usePreferences } from '../../preferences';

export default function TaskStatusCard({
  eyebrow,
  title,
  detail,
  progress = null,
  tone = 'neutral',
}) {
  const { t } = usePreferences();
  const clampedProgress =
    typeof progress === 'number' ? Math.max(0, Math.min(100, Math.round(progress))) : null;

  return (
    <div className={`task-status-card surface-card task-status-${tone}`.trim()}>
      <div className="task-status-copy">
        <span className="status-label">{eyebrow}</span>
        <h4>{title}</h4>
        {detail ? <p>{detail}</p> : null}
      </div>

      {clampedProgress === null ? (
        <div className="typing-indicator" aria-label={title}>
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      ) : (
        <div className="task-status-progress" aria-label={t('progress') || 'Progress'}>
          <div className="task-status-progress-meta">
            <span>{title}</span>
            <strong>{clampedProgress}%</strong>
          </div>
          <div className="task-status-progress-track" aria-hidden="true">
            <div className="task-status-progress-fill" style={{ width: `${clampedProgress}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}
