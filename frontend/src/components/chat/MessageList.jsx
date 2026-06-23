import { useCallback, useEffect, useRef, useState } from 'react';

import TypewriterText from '../common/TypewriterText';
import { usePreferences } from '../../preferences';

export default function MessageList({
  messages,
  emptyLabel,
  animateMessageId = null,
  pendingStage = null,
  streamingMessageId = null,
  streamStats = null,
  speakingMessageId = null,
  onAnimationDone,
  onSpeak,
  onStopSpeaking,
  locale,
}) {
  const listRef = useRef(null);
  const endRef = useRef(null);
  const stickToBottomRef = useRef(true);
  const previousMessageCountRef = useRef(messages.length);
  const preferences = usePreferences();
  const t = preferences.t;
  const dateLocale = locale || preferences.locale;

  const scrollToEnd = useCallback(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, []);

  const updateStickyScroll = useCallback(() => {
    const list = listRef.current;
    if (!list) return;
    stickToBottomRef.current = isNearBottom(list);
  }, []);

  const handleGeneratedTextProgress = useCallback(() => {
    if (!stickToBottomRef.current) return;
    window.requestAnimationFrame(scrollToEnd);
  }, [scrollToEnd]);

  useEffect(() => {
    const previousMessageCount = previousMessageCountRef.current;
    if (messages.length === 0 || messages.length < previousMessageCount) {
      stickToBottomRef.current = true;
    }
    previousMessageCountRef.current = messages.length;
  }, [messages.length]);

  const streamingMessage = streamingMessageId
    ? messages.find((message) => message.id === streamingMessageId)
    : null;
  const streamingContentLength = streamingMessage?.content?.length || 0;
  const streamingReasoningLength = streamingMessage?.payload?.reasoning?.length || 0;

  useEffect(() => {
    if (!stickToBottomRef.current) return undefined;
    const frame = window.requestAnimationFrame(scrollToEnd);
    return () => window.cancelAnimationFrame(frame);
  }, [
    messages.length,
    pendingStage,
    animateMessageId,
    streamingContentLength,
    streamingReasoningLength,
    scrollToEnd,
  ]);

  if (!messages.length) {
    return <ChatWelcome label={emptyLabel} t={t} />;
  }

  return (
    <div className="message-list" ref={listRef} onScroll={updateStickyScroll}>
      {messages.map((message) => {
        const isAssistant = message.role === 'assistant';
        const isSpeaking = speakingMessageId === message.id;
        const isStreaming = streamingMessageId && message.id === streamingMessageId;
        const hasContent = Boolean(message.content?.trim());

        // Show the initial "Analyzing..." placeholder bubble until any
        // content/reasoning has started streaming.
        if (isStreaming && !hasContent && !message.payload?.reasoning) {
          return (
            <StreamingPlaceholderBubble
              key={message.id}
              t={t}
              pendingStage={pendingStage}
              streamStats={streamStats}
            />
          );
        }

        return (
          <article
            key={message.id}
            className={`message-bubble message-${message.role} ${
              isAssistant && !isStreaming ? 'message-bubble-clickable' : ''
            } ${isStreaming ? 'message-streaming-active' : ''}`.trim()}
            onClick={
              isAssistant && !isStreaming
                ? () => (isSpeaking ? onStopSpeaking?.() : onSpeak?.(message))
                : undefined
            }
          >
            <header className="message-meta">
              <div className="message-meta-row">
                <span className="message-meta-author">
                  {isAssistant ? t('assistant') : t('you')}
                </span>
                {isAssistant && !isStreaming ? (
                  <button
                    type="button"
                    className={`message-tool-button ${isSpeaking ? 'message-tool-button-active' : ''}`.trim()}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (isSpeaking) onStopSpeaking?.();
                      else onSpeak?.(message);
                    }}
                    title={isSpeaking ? t('stopReading') : t('readAloud')}
                    aria-label={isSpeaking ? t('stopReadingMessage') : t('readThisMessage')}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true" className="message-tool-icon">
                      {isSpeaking ? (
                        <path d="M7 7h10v10H7z" fill="currentColor" />
                      ) : (
                        <path
                          d="M5 10v4h3l4 4V6L8 10H5Zm10.5 2a3.5 3.5 0 0 0-2-3.15v6.3a3.5 3.5 0 0 0 2-3.15Zm0-7.35v2.1a7 7 0 0 1 0 10.5v2.1a9 9 0 0 0 0-14.7Z"
                          fill="currentColor"
                        />
                      )}
                    </svg>
                  </button>
                ) : null}
              </div>
              <time>
                {isStreaming
                  ? t(pendingStage === 'reasoning' ? 'thinking' : 'streaming')
                  : new Date(message.created_at).toLocaleString(dateLocale)}
              </time>
            </header>

            {isAssistant && message.payload?.reasoning ? (
              <ReasoningPanel
                text={message.payload.reasoning}
                streaming={Boolean(isStreaming && (!hasContent || pendingStage === 'reasoning'))}
                t={t}
              />
            ) : null}

            {hasContent ? (
              <TypewriterText
                text={message.content}
                animate={isAssistant && !isStreaming && message.id === animateMessageId}
                markdown={isAssistant}
                className={isAssistant ? 'message-rich-text' : 'message-plain-text'}
                streaming={Boolean(isStreaming)}
                onProgress={
                  isAssistant && !isStreaming && message.id === animateMessageId
                    ? handleGeneratedTextProgress
                    : undefined
                }
                onDone={message.id === animateMessageId ? onAnimationDone : undefined}
              />
            ) : null}

            {isStreaming ? (
              <GenerationStatsRow stats={streamStats} pendingStage={pendingStage} t={t} />
            ) : null}

            {message.payload?.attachments?.length ? (
              <AttachmentRow message={message} t={t} />
            ) : null}

            {message.payload?.source_names?.length ? (
              <footer className="message-footer-sources">
                {t('sources')}: {message.payload.source_names.join(', ')}
              </footer>
            ) : null}

            {message.payload?.stopped ? (
              <footer className="message-stopped-note">{t('stoppedByUser') || 'Stopped by you.'}</footer>
            ) : null}
          </article>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

function StreamingPlaceholderBubble({ t, pendingStage, streamStats }) {
  let label;
  if (pendingStage === 'reasoning') {
    label = t('thinking') || 'Thinking...';
  } else if (pendingStage === 'streaming') {
    label = t('generating') || 'Generating...';
  } else {
    label = t('analyzingPrompt') || 'Analyzing prompt...';
  }
  return (
    <article className="message-bubble message-assistant message-streaming message-streaming-active">
      <header className="message-meta">
        <div className="message-meta-row">
          <span className="message-meta-author">{t('assistant')}</span>
        </div>
        <time>{label}</time>
      </header>
      <div className="typing-indicator" aria-label={label}>
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
      <GenerationStatsRow stats={streamStats} pendingStage={pendingStage} t={t} />
    </article>
  );
}

function GenerationStatsRow({ stats, pendingStage, t }) {
  const tokens = stats?.tokens;
  const tps = stats?.tokens_per_second;
  const elapsed = stats?.elapsed_seconds;
  if (!stats && !pendingStage) return null;
  return (
    <div className="generation-stats" aria-live="polite">
      {pendingStage === 'streaming' || tokens ? (
        <>
          {typeof tokens === 'number' ? (
            <span className="generation-stat">
              <span className="generation-stat-label">{t('tokens') || 'tokens'}</span>
              <span className="generation-stat-value">{tokens}</span>
            </span>
          ) : null}
          {typeof tps === 'number' && tps > 0 ? (
            <span className="generation-stat">
              <span className="generation-stat-label">tok/s</span>
              <span className="generation-stat-value">{tps.toFixed(1)}</span>
            </span>
          ) : null}
          {typeof elapsed === 'number' ? (
            <span className="generation-stat">
              <span className="generation-stat-label">{t('elapsed') || 'elapsed'}</span>
              <span className="generation-stat-value">{elapsed.toFixed(1)}s</span>
            </span>
          ) : null}
        </>
      ) : (
        <span className="generation-stat generation-stat-muted">
          {pendingStage === 'reasoning'
            ? t('thinking') || 'Thinking...'
            : t('analyzingPrompt') || 'Analyzing prompt...'}
        </span>
      )}
    </div>
  );
}

function ReasoningPanel({ text, streaming, t }) {
  const [open, setOpen] = useState(Boolean(streaming));
  const contentRef = useRef(null);

  // Reflect the parent's "streaming reasoning" state by auto-opening the panel
  // while reasoning tokens are still arriving.
  useEffect(() => {
    if (streaming) setOpen(true);
  }, [streaming]);

  useEffect(() => {
    if (open && streaming && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [text, open, streaming]);

  return (
    <details
      className={`reasoning-panel ${streaming ? 'reasoning-panel-streaming' : ''}`.trim()}
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <span className="reasoning-panel-bullet" aria-hidden="true" />
        <span className="reasoning-panel-label">
          {streaming
            ? t('modelThinking') || 'Model is thinking...'
            : t('modelReasoning') || 'Model reasoning'}
        </span>
        <svg viewBox="0 0 24 24" aria-hidden="true" className="reasoning-panel-caret">
          <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </summary>
      <div className="reasoning-panel-body" ref={contentRef}>
        {text}
        {streaming ? <span className="typewriter-caret reasoning-panel-caret-cursor" aria-hidden="true" /> : null}
      </div>
    </details>
  );
}

function AttachmentRow({ message, t }) {
  return (
    <div className="message-attachment-list">
      {message.payload.attachments.map((attachment, index) => {
        const target = getAttachmentTarget(attachment);
        const kind = String(attachment.kind || '').toLowerCase();
        const previewUrl = kind === 'image' ? target : '';
        const clickable = Boolean(target);
        const key = `${message.id}-attachment-${index}`;

        if (previewUrl) {
          return (
            <button
              key={key}
              type="button"
              className="attachment-image-tile"
              onClick={(event) => {
                event.stopPropagation();
                window.open(previewUrl, '_blank', 'noopener,noreferrer');
              }}
              title={attachment.name}
              aria-label={t('viewImage', { name: attachment.name }) || attachment.name}
            >
              <img src={previewUrl} alt={attachment.name} loading="lazy" />
            </button>
          );
        }

        return clickable ? (
          <button
            key={key}
            type="button"
            className="attachment-chip attachment-chip-static attachment-chip-link"
            onClick={(event) => {
              event.stopPropagation();
              window.open(target, '_blank', 'noopener,noreferrer');
            }}
            title={attachment.name}
          >
            <AttachmentIcon kind={kind} />
            <span>{attachment.name}</span>
          </button>
        ) : (
          <span key={key} className="attachment-chip attachment-chip-static">
            <AttachmentIcon kind={kind} />
            <span>{attachment.name}</span>
          </span>
        );
      })}
    </div>
  );
}

function AttachmentIcon({ kind }) {
  let path;
  if (kind === 'pdf') {
    path = (
      <path
        d="M7 2.5h7L19 7v13.5a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 6 20.5V4a1.5 1.5 0 0 1 1.5-1.5Zm6.5.75V7H17.5l-4-3.75Z"
        fill="currentColor"
        opacity="0.8"
      />
    );
  } else if (kind === 'image') {
    path = (
      <path
        d="M5 4h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Zm0 2v9.6l3.5-3.5 2.5 2.5 4-4L19 14V6H5Zm9.5 4a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"
        fill="currentColor"
        opacity="0.85"
      />
    );
  } else {
    path = (
      <path
        d="M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Zm6 1.4V8h3.6L13 4.4Z"
        fill="currentColor"
        opacity="0.8"
      />
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="attachment-chip-icon">
      {path}
    </svg>
  );
}

function ChatWelcome({ label, t }) {
  return (
    <div className="chat-welcome">
      <div className="chat-welcome-card">
        <span className="chat-welcome-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path
              d="M12 3.5 14.1 9l5.4.6-4 3.7 1.2 5.3L12 15.9l-4.7 2.7 1.2-5.3-4-3.7L9.9 9 12 3.5Z"
              fill="currentColor"
            />
          </svg>
        </span>
        <h3>{t('welcomeHeading') || 'Local assistant ready when you are.'}</h3>
        <p>{label || t('welcomeBody') || 'Ask anything, drop in an image, or paste a document to get started.'}</p>
        <ul className="chat-welcome-tips">
          <li>{t('welcomeTipText') || 'Tip: ask for code, a comparison, or a story.'}</li>
          <li>{t('welcomeTipImage') || 'Tip: attach a PNG or JPEG to ask about an image.'}</li>
          <li>{t('welcomeTipStop') || 'Tip: hit Stop any time to cancel a long generation.'}</li>
        </ul>
      </div>
    </div>
  );
}

function isNearBottom(element) {
  const bottomGap = element.scrollHeight - element.scrollTop - element.clientHeight;
  return bottomGap < 96;
}

function getAttachmentTarget(attachment) {
  if (!attachment) return '';
  const directTarget = attachment.object_url || attachment.objectUrl || attachment.download_url || attachment.url;
  if (directTarget) return directTarget;
  return toFileUrl(attachment.storage_path) || toFileUrl(attachment.text_path);
}

function toFileUrl(path) {
  if (!path || typeof path !== 'string') return '';
  if (/^(blob:|data:|file:\/\/|https?:\/\/)/i.test(path)) return path;
  const normalized = path.replace(/\\/g, '/');
  if (/^[A-Za-z]:\//.test(normalized)) {
    return `file:///${encodeURI(normalized)}`;
  }
  if (normalized.startsWith('/')) {
    return `file://${encodeURI(normalized)}`;
  }
  return '';
}
