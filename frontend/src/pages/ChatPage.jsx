import { useEffect, useRef, useState } from 'react';

import { listChatMessages, streamChatMessage } from '../api/chat';
import MessageComposer from '../components/chat/MessageComposer';
import MessageList from '../components/chat/MessageList';
import { usePreferences } from '../preferences';
import { useSession } from '../session';

const STREAMING_ID = '__streaming__';

export default function ChatPage() {
  const { locale, speechLanguage, t } = usePreferences();
  const { sessionId, loading, error: sessionError, refreshSession } = useSession();
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const [pendingStage, setPendingStage] = useState(null); // 'analyzing' | 'streaming' | null
  const [streamStats, setStreamStats] = useState(null);
  const [animateMessageId, setAnimateMessageId] = useState(null);
  const [speakingMessageId, setSpeakingMessageId] = useState(null);
  const abortRef = useRef(null);
  const messagesRef = useRef([]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (!sessionId) return;
    listChatMessages(sessionId)
      .then((loadedMessages) => {
        setMessages((current) => {
          revokeMessageObjectUrls(current);
          return loadedMessages;
        });
      })
      .catch((loadError) => setError(loadError.message));
  }, [sessionId]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      revokeMessageObjectUrls(messagesRef.current);
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    },
    [],
  );

  async function handleSend(message, files = []) {
    if (!sessionId) return;

    const localUserId = `local-${Date.now()}`;
    const optimisticText = message || t('attachmentAnalyze');
    const controller = new AbortController();
    const optimisticAttachments = files.map((file) => ({
      name: file.name,
      kind: inferAttachmentKind(file.name),
      object_url: URL.createObjectURL(file),
    }));

    abortRef.current = controller;
    setSending(true);
    setPendingStage('analyzing');
    setStreamStats(null);
    setError('');

    let userMessageId = localUserId;
    let aborted = false;

    setMessages((current) => [
      ...current,
      {
        id: localUserId,
        mode: 'general',
        role: 'user',
        content: optimisticText,
        payload: { attachments: optimisticAttachments },
        created_at: new Date().toISOString(),
      },
      {
        id: STREAMING_ID,
        mode: 'general',
        role: 'assistant',
        content: '',
        payload: {},
        created_at: new Date().toISOString(),
      },
    ]);

    try {
      await streamChatMessage(sessionId, message, files, {
        signal: controller.signal,
        onEvent: (event) => {
          if (!event || typeof event !== 'object') return;
          switch (event.type) {
            case 'analyzing':
              setPendingStage('analyzing');
              break;
            case 'user':
              if (event.message?.id) {
                userMessageId = event.message.id;
                setMessages((current) =>
                  current.map((item) =>
                    item.id === localUserId
                      ? { ...event.message, payload: { ...(event.message.payload || {}), attachments: mergeAttachments(event.message.payload?.attachments, optimisticAttachments) } }
                      : item,
                  ),
                );
              }
              break;
            case 'reasoning_token': {
              const delta = String(event.text || '');
              if (!delta) break;
              setPendingStage((current) => (current === 'streaming' ? current : 'reasoning'));
              setMessages((current) =>
                current.map((item) =>
                  item.id === STREAMING_ID
                    ? {
                        ...item,
                        payload: {
                          ...(item.payload || {}),
                          reasoning: `${item.payload?.reasoning || ''}${delta}`,
                        },
                      }
                    : item,
                ),
              );
              break;
            }
            case 'token': {
              const delta = String(event.text || '');
              if (!delta) break;
              setPendingStage('streaming');
              setMessages((current) =>
                current.map((item) =>
                  item.id === STREAMING_ID ? { ...item, content: `${item.content || ''}${delta}` } : item,
                ),
              );
              break;
            }
            case 'replace': {
              const text = String(event.text || '');
              setMessages((current) =>
                current.map((item) => (item.id === STREAMING_ID ? { ...item, content: text } : item)),
              );
              break;
            }
            case 'meta':
              setStreamStats((current) => ({ ...(current || {}), ...event }));
              break;
            case 'done':
              setStreamStats((current) => ({ ...(current || {}), ...(event.stats || {}) }));
              if (event.message) {
                const persistedReasoning = event.reasoning || event.message?.payload?.reasoning;
                const merged = persistedReasoning
                  ? {
                      ...event.message,
                      payload: { ...(event.message.payload || {}), reasoning: persistedReasoning },
                    }
                  : event.message;
                setMessages((current) =>
                  current.map((item) => (item.id === STREAMING_ID ? merged : item)),
                );
                // Do not animate again - text is already on screen via streaming.
                setAnimateMessageId(null);
              }
              break;
            case 'error':
              throw new Error(event.detail || 'Streaming failed.');
            default:
              break;
          }
        },
      });
      refreshSession?.().catch(() => {});
    } catch (sendError) {
      aborted = sendError?.name === 'AbortError' || controller.signal.aborted;
      if (aborted) {
        // Keep whatever text streamed in so far, but stamp a synthetic id so it
        // does not clash with future server messages.
        setMessages((current) =>
          current
            .map((item) =>
              item.id === STREAMING_ID
                ? { ...item, id: `stopped-${Date.now()}`, payload: { ...(item.payload || {}), stopped: true } }
                : item,
            )
            // Drop empty assistant placeholders if nothing came through.
            .filter((item) => !(item.role === 'assistant' && !item.content?.trim() && item.payload?.stopped)),
        );
      } else {
        revokeAttachmentObjectUrls(optimisticAttachments);
        setMessages((current) => current.filter((item) => item.id !== STREAMING_ID && item.id !== userMessageId && item.id !== localUserId));
        setError(sendError.message || 'Streaming failed.');
      }
    } finally {
      abortRef.current = null;
      setPendingStage(null);
      setSending(false);
    }
  }

  function handleStop() {
    if (!abortRef.current) {
      return;
    }
    abortRef.current.abort();
    abortRef.current = null;
    setPendingStage(null);
    setSending(false);
  }

  function handleSpeak(message) {
    if (!window.speechSynthesis) {
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message.content);
    utterance.lang = speechLanguage;
    utterance.rate = 1;
    utterance.onend = () => setSpeakingMessageId((current) => (current === message.id ? null : current));
    utterance.onerror = () => setSpeakingMessageId((current) => (current === message.id ? null : current));
    setSpeakingMessageId(message.id);
    window.speechSynthesis.speak(utterance);
  }

  function handleStopSpeaking() {
    if (!window.speechSynthesis) {
      return;
    }
    window.speechSynthesis.cancel();
    setSpeakingMessageId(null);
  }

  const empty = messages.length === 0;
  const generating = sending || Boolean(pendingStage);

  return (
    <div className={`chat-page ${empty ? 'chat-page-empty' : ''}`}>
      <div className="page-header compact-header">
        <div>
          <p className="eyebrow">Assistant</p>
          <h2>{t('generalChat')}</h2>
        </div>
      </div>

      {error || sessionError ? <p className="error-text">{error || sessionError}</p> : null}

      <section className={`chat-stage ${empty ? 'chat-stage-empty' : ''}`}>
        {!empty ? (
          <MessageList
            messages={messages}
            emptyLabel={t('noChatHistory')}
            animateMessageId={animateMessageId}
            pendingStage={pendingStage}
            streamingMessageId={STREAMING_ID}
            streamStats={streamStats}
            speakingMessageId={speakingMessageId}
            locale={locale}
            onAnimationDone={() => setAnimateMessageId(null)}
            onSpeak={handleSpeak}
            onStopSpeaking={handleStopSpeaking}
          />
        ) : (
          <div className="chat-stage-spacer" aria-hidden="true" />
        )}

        <MessageComposer
          sessionId={sessionId}
          disabled={loading || !sessionId}
          pending={generating}
          onSend={handleSend}
          onStop={handleStop}
        />
      </section>
    </div>
  );
}

function inferAttachmentKind(fileName) {
  const lower = fileName.toLowerCase();
  if (lower.endsWith('.pdf')) return 'pdf';
  if (/\.(png|jpe?g|webp|bmp|gif)$/.test(lower)) return 'image';
  return 'text';
}

function revokeMessageObjectUrls(messageList) {
  messageList.forEach((message) => {
    revokeAttachmentObjectUrls(message.payload?.attachments || []);
  });
}

function revokeAttachmentObjectUrls(attachments) {
  attachments.forEach((attachment) => {
    if (attachment?.object_url) {
      URL.revokeObjectURL(attachment.object_url);
    }
  });
}

function mergeAttachments(serverAttachments = [], optimistic = []) {
  if (!serverAttachments.length) return optimistic;
  return serverAttachments.map((attachment, index) => ({
    ...attachment,
    object_url: attachment.object_url || optimistic[index]?.object_url,
  }));
}
