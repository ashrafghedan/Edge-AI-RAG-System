import TypewriterText from '../common/TypewriterText';
import { usePreferences } from '../../preferences';
import TaskStatusCard from './TaskStatusCard';

export default function GradingCard({ grading, loading = false }) {
  const { t } = usePreferences();

  if (loading) {
    return (
      <TaskStatusCard
        eyebrow={t('readingWorkflow')}
        title={t('gradingInProgress')}
        detail={t('gradingInProgressHelp')}
      />
    );
  }

  if (!grading) {
    return <div className="empty-panel">{t('noGrading')}</div>;
  }

  return (
    <section className="result-card surface-card">
      <div className="score-block">
        <span className="status-label">{t('score')}</span>
        <h3>{grading.score}/10</h3>
      </div>
      <div>
        <span className="status-label">{t('feedback')}</span>
        <TypewriterText key={`feedback-${grading.attempt_id}`} text={grading.feedback} animate />
      </div>
      <div>
        <span className="status-label">{t('modelAnswer')}</span>
        <TypewriterText key={`model-answer-${grading.attempt_id}`} text={grading.model_answer} animate />
      </div>
    </section>
  );
}
