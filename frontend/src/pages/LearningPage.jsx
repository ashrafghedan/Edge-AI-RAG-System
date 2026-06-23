import { useEffect, useMemo, useState } from 'react';

import {
  askGroundedQuestion,
  generateQuestion,
  gradeQuestion,
  listAttempts,
  listGeneratedQuestions,
} from '../api/learning';
import SpeechToTextButton from '../components/common/SpeechToTextButton';
import TypewriterText from '../components/common/TypewriterText';
import DocumentManager from '../components/learning/DocumentManager';
import GradingCard from '../components/learning/GradingCard';
import TaskStatusCard from '../components/learning/TaskStatusCard';
import { usePreferences } from '../preferences';
import { useSession } from '../session';

export default function LearningPage() {
  const { locale, t } = usePreferences();
  const { sessionId, refreshSession, error: sessionError } = useSession();
  const [activeCorpus, setActiveCorpus] = useState(null);
  const [groundedQuestion, setGroundedQuestion] = useState('');
  const [groundedAnswer, setGroundedAnswer] = useState(null);
  const [generatedQuestion, setGeneratedQuestion] = useState(null);
  const [generatedQuestions, setGeneratedQuestions] = useState([]);
  const [userAnswer, setUserAnswer] = useState('');
  const [grading, setGrading] = useState(null);
  const [attempts, setAttempts] = useState([]);
  const [error, setError] = useState('');
  const [activeTask, setActiveTask] = useState(null);
  const [groundedAnimationKey, setGroundedAnimationKey] = useState(0);
  const [questionAnimationKey, setQuestionAnimationKey] = useState(0);
  const busy = activeTask !== null;

  const latestAttemptByQuestion = useMemo(() => {
    const mapping = new Map();
    for (const attempt of attempts) {
      if (!mapping.has(attempt.question_id)) {
        mapping.set(attempt.question_id, attempt);
      }
    }
    return mapping;
  }, [attempts]);

  async function refreshLearningData() {
    if (!sessionId) return;
    const [session, questions, history] = await Promise.all([
      refreshSession(),
      listGeneratedQuestions(sessionId),
      listAttempts(sessionId),
    ]);
    setActiveCorpus(session?.active_corpus || null);
    setGeneratedQuestions(questions);
    setAttempts(history);
  }

  useEffect(() => {
    if (!sessionId) return;
    refreshLearningData().catch((loadError) => setError(loadError.message));
  }, [sessionId]);

  const handleAsk = async () => {
    if (!groundedQuestion.trim()) return;
    setActiveTask('ask');
    setError('');
    try {
      const result = await askGroundedQuestion(sessionId, groundedQuestion);
      setGroundedAnswer(result);
      setGroundedAnimationKey((value) => value + 1);
    } catch (askError) {
      setError(askError.message);
    } finally {
      setActiveTask(null);
    }
  };

  const handleGenerate = async () => {
    setActiveTask('generate');
    setError('');
    try {
      const result = await generateQuestion(sessionId);
      setGeneratedQuestion(result);
      setQuestionAnimationKey((value) => value + 1);
      setUserAnswer('');
      setGrading(null);
      await refreshLearningData();
    } catch (generateError) {
      setError(generateError.message);
    } finally {
      setActiveTask(null);
    }
  };

  const handleGrade = async () => {
    if (!generatedQuestion || !userAnswer.trim()) return;
    setActiveTask('grade');
    setError('');
    try {
      const result = await gradeQuestion(sessionId, generatedQuestion.question_id, userAnswer);
      setGrading(result);
      await refreshLearningData();
    } catch (gradeError) {
      setError(gradeError.message);
    } finally {
      setActiveTask(null);
    }
  };

  return (
    <div className="page learning-page">
      <div className="page-header page-header-minimal">
        <p className="eyebrow">{t('ragStudio')}</p>
      </div>

      <div className="page-scroll">
        <DocumentManager
          sessionId={sessionId}
          onCorpusChange={async (corpus) => {
            setActiveCorpus(corpus);
            await refreshLearningData();
          }}
        />

        {error || sessionError ? <p className="error-text">{error || sessionError}</p> : null}

        <div className="learning-grid">
          <section className="panel workspace-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t('groundedQa')}</p>
                <h3>{t('askAboutActiveText')}</h3>
              </div>
              <span className="status-chip">{activeCorpus?.dataset_label || t('noCorpusSelected')}</span>
            </div>

            <div className="field-stack">
              <label className="field-label" htmlFor="grounded-question">
                {t('question')}
              </label>
              <textarea
                id="grounded-question"
                rows={5}
                value={groundedQuestion}
                onChange={(event) => setGroundedQuestion(event.target.value)}
                placeholder={t('groundedQuestionPlaceholder')}
              />
            </div>
            <div className="action-row action-row-spread panel-actions">
              <button
                type="button"
                className="action-button"
                onClick={handleAsk}
                disabled={busy || !activeCorpus || !groundedQuestion.trim()}
              >
                {t('askGroundedQuestion')}
              </button>
              <SpeechToTextButton
                sessionId={sessionId}
                value={groundedQuestion}
                onValueChange={setGroundedQuestion}
                disabled={busy || !activeCorpus}
              />
            </div>

            {activeTask === 'ask' ? (
              <TaskStatusCard
                eyebrow={t('groundedQa')}
                title={t('searchingActiveCorpus')}
                detail={t('searchingActiveCorpusHelp')}
              />
            ) : groundedAnswer ? (
              <div className="answer-card surface-card spaced-card">
                <div className="answer-status-row">
                  <span
                    className={`answer-state ${
                      groundedAnswer.found ? 'answer-state-found' : 'answer-state-missing'
                    }`}
                  >
                    {groundedAnswer.found ? t('groundedAnswerFound') : t('notAvailableInCorpus')}
                  </span>
                  <span className="muted-inline">
                    {t('sources')}: {groundedAnswer.source_names.join(', ') || t('none')}
                  </span>
                </div>
                <TypewriterText
                  key={`grounded-${groundedAnimationKey}`}
                  text={groundedAnswer.answer}
                  animate
                />
              </div>
            ) : (
              <div className="empty-panel">{t('noGroundedAnswer')}</div>
            )}
          </section>

          <section className="panel workspace-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t('readingWorkflow')}</p>
                <h3>{t('generateAndGrade')}</h3>
              </div>
            </div>

            <div className="action-row panel-actions panel-actions-top">
              <button
                type="button"
                className="action-button"
                onClick={handleGenerate}
                disabled={busy || !activeCorpus}
              >
                {t('generateQuestion')}
              </button>
            </div>

            {activeTask === 'generate' ? (
              <TaskStatusCard
                eyebrow={t('readingWorkflow')}
                title={t('generatingStudyQuestion')}
                detail={t('generatingStudyQuestionHelp')}
              />
            ) : generatedQuestion ? (
              <div className="question-card surface-card spaced-card">
                <div className="question-header-row">
                  <span className="status-label">{t('currentQuestion')}</span>
                </div>
                <TypewriterText
                  key={`question-${generatedQuestion.question_id}-${questionAnimationKey}`}
                  text={generatedQuestion.question}
                  animate
                  as="div"
                  className="question-text"
                />

                <div className="field-stack">
                  <div className="field-toolbar">
                    <label className="field-label" htmlFor="generated-answer">
                      {t('yourAnswer')}
                    </label>
                    <SpeechToTextButton
                      sessionId={sessionId}
                      value={userAnswer}
                      onValueChange={setUserAnswer}
                      disabled={busy}
                    />
                  </div>
                  <textarea
                    id="generated-answer"
                    rows={6}
                    value={userAnswer}
                    onChange={(event) => setUserAnswer(event.target.value)}
                    placeholder={t('generatedAnswerPlaceholder')}
                  />
                </div>

                <div className="action-row panel-actions">
                  <button
                    type="button"
                    className="action-button"
                    onClick={handleGrade}
                    disabled={busy || !userAnswer.trim()}
                  >
                    {t('gradeAnswer')}
                  </button>
                </div>
              </div>
            ) : (
              <div className="empty-panel">{t('noGeneratedQuestion')}</div>
            )}

            <GradingCard grading={grading} loading={activeTask === 'grade'} />
          </section>
        </div>

        <section className="panel history-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">{t('history')}</p>
              <h3>{t('questionsAndStatus')}</h3>
            </div>
          </div>
          <div className="history-grid">
            <div className="surface-card history-column">
              <h4>{t('generatedQuestions')}</h4>
              <ul className="history-list">
                {generatedQuestions.length ? (
                  generatedQuestions.map((question) => (
                    <li key={question.question_id}>
                      <strong>{question.question}</strong>
                      <span>{new Date(question.created_at).toLocaleString(locale)}</span>
                    </li>
                  ))
                ) : (
                  <li className="history-empty">{t('noGeneratedQuestions')}</li>
                )}
              </ul>
            </div>
            <div className="surface-card history-column">
              <h4>{t('answerStatus')}</h4>
              <ul className="history-list">
                {generatedQuestions.length ? (
                  generatedQuestions.map((question) => {
                    const latestAttempt = latestAttemptByQuestion.get(question.question_id);
                    return (
                      <li key={`status-${question.question_id}`}>
                        <strong>
                          {latestAttempt ? `${latestAttempt.score}/10` : t('awaitingAnswer')}
                        </strong>
                        <span>
                          {latestAttempt ? latestAttempt.feedback : t('thisQuestionNotAnswered')}
                        </span>
                      </li>
                    );
                  })
                ) : (
                  <li className="history-empty">{t('noQuestionStatus')}</li>
                )}
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
