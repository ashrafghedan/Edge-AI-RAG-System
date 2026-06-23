import { useEffect, useMemo, useRef, useState } from 'react';

import SpeechToTextButton from '../common/SpeechToTextButton';
import { usePreferences } from '../../preferences';

const ACCEPTED_ATTACHMENTS = '.txt,.pdf,.png,.jpg,.jpeg,.webp,.bmp,.gif';
const VALID_ATTACHMENT_PATTERN = /\.(txt|pdf|png|jpe?g|webp|bmp|gif)$/i;
const IMAGE_PATTERN = /\.(png|jpe?g|webp|bmp|gif)$/i;
const MAX_FILE_BYTES = 12 * 1024 * 1024;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const MAX_FILES = 6;

export default function MessageComposer({ sessionId, disabled, pending = false, onSend, onStop }) {
  const { t } = usePreferences();
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [speechError, setSpeechError] = useState('');
  const [uploadError, setUploadError] = useState('');
  const inputRef = useRef(null);
  const textareaRef = useRef(null);
  const dragDepthRef = useRef(0);
  const fileKeysRef = useRef(new WeakMap());
  const previewUrlsRef = useRef(new WeakMap());

  const hasContent = Boolean(message.trim() || files.length);
  const canInteract = !disabled && !pending;

  const previews = useMemo(
    () =>
      files.map((file) => {
        const isImage = IMAGE_PATTERN.test(file.name || '');
        let previewUrl = previewUrlsRef.current.get(file);
        if (isImage && !previewUrl) {
          previewUrl = URL.createObjectURL(file);
          previewUrlsRef.current.set(file, previewUrl);
        }
        return {
          file,
          key: keyFor(file, fileKeysRef.current),
          isImage,
          previewUrl: isImage ? previewUrl : '',
          size: formatBytes(file.size || 0),
        };
      }),
    [files],
  );

  // Auto-grow the textarea up to a reasonable cap so the composer feels less
  // cramped on long pastes.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = 'auto';
    const next = Math.min(node.scrollHeight, 280);
    node.style.height = `${next}px`;
  }, [message]);

  // Revoke object URLs once a file is dropped or the composer unmounts.
  useEffect(
    () => () => {
      previews.forEach((preview) => {
        if (preview.previewUrl) URL.revokeObjectURL(preview.previewUrl);
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!hasContent || pending) return;
    const trimmed = message.trim();
    const filesToSend = files;
    await onSend(trimmed, filesToSend);
    filesToSend.forEach((file) => {
      const url = previewUrlsRef.current.get(file);
      if (url) URL.revokeObjectURL(url);
      previewUrlsRef.current.delete(file);
    });
    setMessage('');
    setFiles([]);
    setDragActive(false);
    setSpeechError('');
    setUploadError('');
    if (inputRef.current) inputRef.current.value = '';
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = async (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      await handleSubmit(event);
    }
  };

  const mergeFiles = (pickedFiles) => {
    setUploadError('');
    const picked = Array.from(pickedFiles || []);
    if (!picked.length) return;

    const accepted = [];
    const errors = [];
    for (const file of picked) {
      if (!VALID_ATTACHMENT_PATTERN.test(file.name || '')) {
        errors.push(t('rejectedFileType', { name: file.name }) || `${file.name}: unsupported file type.`);
        continue;
      }
      const limit = IMAGE_PATTERN.test(file.name) ? MAX_IMAGE_BYTES : MAX_FILE_BYTES;
      if (file.size > limit) {
        errors.push(
          t('rejectedFileSize', { name: file.name, limit: formatBytes(limit) })
            || `${file.name}: larger than ${formatBytes(limit)}.`,
        );
        continue;
      }
      accepted.push(file);
    }

    if (!accepted.length) {
      if (errors.length) setUploadError(errors.join(' '));
      return;
    }

    setFiles((current) => {
      const seen = new Set(current.map((file) => keyFor(file, fileKeysRef.current)));
      const merged = [...current];
      for (const file of accepted) {
        const key = keyFor(file, fileKeysRef.current);
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(file);
      }
      if (merged.length > MAX_FILES) {
        errors.push(t('tooManyFiles', { count: MAX_FILES }) || `Only ${MAX_FILES} attachments per message.`);
        return merged.slice(0, MAX_FILES);
      }
      return merged;
    });

    if (errors.length) setUploadError(errors.join(' '));
  };

  const handleFilePick = (event) => {
    mergeFiles(event.target.files);
    if (inputRef.current) inputRef.current.value = '';
  };

  const removeFile = (target) => {
    const url = previewUrlsRef.current.get(target);
    if (url) {
      URL.revokeObjectURL(url);
      previewUrlsRef.current.delete(target);
    }
    setFiles((current) => current.filter((file) => file !== target));
  };

  const handleDragEnter = (event) => {
    event.preventDefault();
    if (!canInteract) return;
    dragDepthRef.current += 1;
    setDragActive(true);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    if (!canInteract) return;
    event.dataTransfer.dropEffect = 'copy';
    setDragActive(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    if (!canInteract) return;
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0 || !event.currentTarget.contains(event.relatedTarget)) {
      setDragActive(false);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    dragDepthRef.current = 0;
    setDragActive(false);
    if (!canInteract) return;
    mergeFiles(event.dataTransfer?.files);
  };

  const handlePaste = (event) => {
    if (!canInteract) return;
    const pastedFiles = Array.from(event.clipboardData?.files || []);
    if (!pastedFiles.length) return;
    event.preventDefault();
    mergeFiles(pastedFiles);
  };

  return (
    <form className="composer composer-dock" onSubmit={handleSubmit}>
      <div
        className={`composer-shell ${dragActive ? 'composer-shell-drop-active' : ''}`.trim()}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {previews.length ? (
          <div className="composer-attachment-row" aria-label={t('selectedFiles') || 'Selected files'}>
            {previews.map((preview) => (
              <div
                key={preview.key}
                className={`composer-attachment-card ${preview.isImage ? 'composer-attachment-image' : ''}`.trim()}
              >
                {preview.isImage ? (
                  <img src={preview.previewUrl} alt={preview.file.name} loading="lazy" />
                ) : (
                  <div className="composer-attachment-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24">
                      <path
                        d="M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Zm6 1.4V8h3.6L13 4.4Z"
                        fill="currentColor"
                      />
                    </svg>
                  </div>
                )}
                <div className="composer-attachment-meta">
                  <span className="composer-attachment-name">{preview.file.name}</span>
                  <span className="composer-attachment-size">{preview.size}</span>
                </div>
                <button
                  type="button"
                  className="composer-attachment-remove"
                  onClick={() => removeFile(preview.file)}
                  disabled={!canInteract}
                  aria-label={t('removeFile', { name: preview.file.name }) || `Remove ${preview.file.name}`}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path
                      d="M7 7l10 10M17 7L7 17"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      fill="none"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        ) : null}

        {dragActive ? <div className="composer-drop-indicator">{t('dropFiles')}</div> : null}

        {uploadError ? <p className="composer-error">{uploadError}</p> : null}
        {speechError ? <p className="composer-error">{speechError}</p> : null}

        <textarea
          ref={textareaRef}
          value={message}
          onChange={(event) => {
            setMessage(event.target.value);
            if (speechError) setSpeechError('');
            if (uploadError) setUploadError('');
          }}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={dragActive ? t('dropFilesPlaceholder') : t('messageAssistant')}
          rows={1}
          disabled={disabled || pending}
          className="composer-input"
        />

        <div className="composer-actions">
          <div className="composer-actions-left">
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_ATTACHMENTS}
              multiple
              onChange={handleFilePick}
              disabled={!canInteract}
              className="composer-file-input"
            />
            <button
              type="button"
              className="composer-icon-button"
              title={t('attachFiles')}
              onClick={() => inputRef.current?.click()}
              disabled={!canInteract}
              aria-label={t('attachFiles')}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true" className="composer-icon-svg">
                <path
                  d="M9 11.5V7a3 3 0 1 1 6 0v9a5 5 0 1 1-10 0V8.5"
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="1.8"
                />
              </svg>
            </button>
            <SpeechToTextButton
              sessionId={sessionId}
              value={message}
              onValueChange={setMessage}
              onError={setSpeechError}
              disabled={!canInteract}
            />
          </div>
          <div className="composer-actions-right">
            {pending ? (
              <button type="button" className="stop-button" onClick={onStop} disabled={disabled}>
                <span className="stop-icon" aria-hidden="true" />
                {t('stop')}
              </button>
            ) : (
              <button type="submit" className="send-button" disabled={disabled || !hasContent}>
                <span>{t('send')}</span>
                <svg viewBox="0 0 24 24" aria-hidden="true" className="send-icon">
                  <path d="M4 20l16-8L4 4l3 8-3 8Zm5-6l8-2-8-2v4Z" fill="currentColor" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </form>
  );
}

function keyFor(file, map) {
  let existing = map.get(file);
  if (existing) return existing;
  existing = `${file.name}:${file.size}:${file.lastModified}:${Math.random().toString(36).slice(2, 8)}`;
  map.set(file, existing);
  return existing;
}

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}
