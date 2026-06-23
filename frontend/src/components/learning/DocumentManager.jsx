import { useEffect, useMemo, useState } from 'react';

import {
  activateCorpus,
  deleteDocument,
  getActiveCorpus,
  listDocuments,
  uploadDocuments,
} from '../../api/learning';
import { usePreferences } from '../../preferences';
import TaskStatusCard from './TaskStatusCard';

export default function DocumentManager({ sessionId, onCorpusChange }) {
  const { locale, t } = usePreferences();
  const [documents, setDocuments] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [activeCorpus, setActiveCorpus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [activationState, setActivationState] = useState(null);
  const [uploadState, setUploadState] = useState(null);

  const hasSelection = useMemo(() => selectedIds.length > 0, [selectedIds]);
  const selectedDocuments = useMemo(
    () => documents.filter((document) => selectedIds.includes(document.id)),
    [documents, selectedIds],
  );
  const selectedReadyCount = useMemo(
    () => selectedDocuments.filter((document) => document.chunk_cache_ready).length,
    [selectedDocuments],
  );
  const selectedCachedChunkCount = useMemo(
    () => selectedDocuments.reduce((total, document) => total + (document.cached_chunk_count || 0), 0),
    [selectedDocuments],
  );

  async function refresh() {
    const docs = await listDocuments(sessionId);
    setDocuments(docs);
    try {
      const active = await getActiveCorpus(sessionId);
      setActiveCorpus(active);
      onCorpusChange?.(active);
      setSelectedIds((current) => {
        const validCurrent = current.filter((value) => docs.some((document) => document.id === value));
        if (validCurrent.length) return validCurrent;
        return active?.document_ids || [];
      });
    } catch (_error) {
      setActiveCorpus(null);
      onCorpusChange?.(null);
      setSelectedIds((current) => current.filter((value) => docs.some((document) => document.id === value)));
    }
  }

  useEffect(() => {
    if (!sessionId) return;
    refresh().catch((loadError) => setError(loadError.message));
  }, [sessionId]);

  useEffect(() => {
    if (!activationState?.running) return undefined;

    const timer = window.setInterval(() => {
      setActivationState((current) => {
        if (!current?.running) return current;
        const elapsed = (Date.now() - current.startedAt) / 1000;
        const stage = activationStage(elapsed, current, t);
        const nextProgress = Math.min(
          stage.progress,
          current.progress + Math.max(1, (stage.progress - current.progress) * 0.22),
        );
        return {
          ...current,
          progress: Math.min(95, nextProgress),
          title: stage.title,
          detail: stage.detail,
        };
      });
    }, 220);

    return () => window.clearInterval(timer);
  }, [activationState?.running, t]);

  useEffect(() => {
    if (uploadState?.phase !== 'processing') return undefined;

    const timer = window.setInterval(() => {
      setUploadState((current) => {
        if (current?.phase !== 'processing') return current;
        return {
          ...current,
          progress: Math.min(98, current.progress + Math.max(1, (98 - current.progress) * 0.12)),
        };
      });
    }, 280);

    return () => window.clearInterval(timer);
  }, [uploadState?.phase]);

  const handleUpload = async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length || !sessionId) return;
    setBusy(true);
    setError('');
    const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
    setUploadState({
      phase: 'uploading',
      progress: 0,
      fileCount: files.length,
      totalBytes,
      title: t('uploadingDocuments'),
      detail: t('uploadingDocumentsHelp', { count: files.length }),
    });
    try {
      await uploadDocuments(sessionId, files, {
        onUploadProgress: ({ progress }) => {
          setUploadState((current) => ({
            ...(current || {}),
            phase: 'uploading',
            progress: Math.max(1, Math.min(90, progress * 0.9)),
            fileCount: files.length,
            totalBytes,
            title: t('uploadingDocuments'),
            detail: t('uploadingDocumentsHelp', { count: files.length }),
          }));
        },
        onProcessingStart: () => {
          setUploadState({
            phase: 'processing',
            progress: 92,
            fileCount: files.length,
            totalBytes,
            title: t('processingUpload'),
            detail: t('processingUploadHelp'),
          });
        },
      });
      await refresh();
      setUploadState({
        phase: 'done',
        progress: 100,
        fileCount: files.length,
        totalBytes,
        title: t('uploadComplete'),
        detail: t('filesCount', { count: files.length }),
      });
      window.setTimeout(() => {
        setUploadState((current) => (current?.phase === 'done' ? null : current));
      }, 1200);
    } catch (uploadError) {
      setUploadState(null);
      setError(uploadError.message);
    } finally {
      setBusy(false);
      event.target.value = '';
    }
  };

  const handleActivate = async () => {
    if (!sessionId) return;
    setBusy(true);
    setError('');
    setActivationState({
      running: true,
      startedAt: Date.now(),
      progress: 4,
      documentCount: selectedDocuments.length,
      readyCount: selectedReadyCount,
      cachedChunkCount: selectedCachedChunkCount,
      allCached: selectedDocuments.length > 0 && selectedReadyCount === selectedDocuments.length,
      title: t('preparingCorpus'),
      detail: activationIntroDetail(
        {
          documentCount: selectedDocuments.length,
          readyCount: selectedReadyCount,
          cachedChunkCount: selectedCachedChunkCount,
          allCached: selectedDocuments.length > 0 && selectedReadyCount === selectedDocuments.length,
        },
        t,
      ),
    });
    try {
      const active = await activateCorpus(sessionId, selectedIds);
      setActiveCorpus(active);
      setSelectedIds(active.document_ids || selectedIds);
      onCorpusChange?.(active);
      setActivationState({
        running: false,
        progress: 100,
        title: t('corpusReady'),
        detail: t('chunksReady', { count: active.chunk_count }),
      });
      window.setTimeout(() => {
        setActivationState((current) => (current?.running ? current : null));
      }, 1400);
      await refresh();
    } catch (activateError) {
      setActivationState(null);
      setError(activateError.message);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (documentId, originalName) => {
    if (!sessionId) return;
    const confirmed = window.confirm(t('confirmDelete', { name: originalName }));
    if (!confirmed) return;
    setBusy(true);
    setError('');
    try {
      await deleteDocument(sessionId, documentId);
      setSelectedIds((current) => current.filter((value) => value !== documentId));
      await refresh();
    } catch (deleteError) {
      setError(deleteError.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel document-panel">
      <div className="panel-header panel-header-tight">
        <p className="eyebrow">{t('documents')}</p>
        <label className="upload-button">
          <input
            type="file"
            accept=".txt,.pdf,application/pdf,text/plain"
            multiple
            onChange={handleUpload}
            disabled={busy || !sessionId}
          />
          {uploadState?.phase === 'uploading' || uploadState?.phase === 'processing'
            ? t('pleaseWait')
            : t('uploadFile')}
        </label>
      </div>

      {error ? <p className="error-text">{error}</p> : null}
      {uploadState ? (
        <TaskStatusCard
          eyebrow={t('documents')}
          title={uploadState.title}
          detail={uploadState.detail}
          progress={uploadState.progress}
        />
      ) : null}

      <div className="document-grid">
        <div className="document-list surface-card spaced-card">
          <div className="list-header-row">
            <span className="status-label">{t('availableDocuments')}</span>
            <span className="muted-inline">{t('filesCount', { count: documents.length })}</span>
          </div>
          {documents.length ? (
            documents.map((document) => (
              <div key={document.id} className="document-entry">
                <label
                  className={`document-row ${document.is_active ? 'document-row-active' : ''} ${
                    document.chunk_cache_ready ? 'document-row-ready' : ''
                  }`.trim()}
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(document.id)}
                    onChange={(event) => {
                      setSelectedIds((current) =>
                        event.target.checked
                          ? [...current, document.id]
                          : current.filter((value) => value !== document.id),
                      );
                    }}
                  />
                  <div className="document-row-body">
                    <strong>{document.original_name}</strong>
                    <div className="document-row-statuses">
                      {document.is_active ? (
                        <span className="document-state-badge document-state-active">{t('activeNow')}</span>
                      ) : null}
                      {document.chunk_cache_ready ? (
                        <span className="document-state-badge document-state-ready">
                          {document.cached_chunk_count
                            ? t('cachedChunksCount', { count: document.cached_chunk_count })
                            : t('cachedChunksReady')}
                        </span>
                      ) : (
                        <span className="document-state-badge document-state-pending">{t('needsChunking')}</span>
                      )}
                    </div>
                    <span>
                      {(document.size_bytes / 1024).toFixed(1)} KB /{' '}
                      {new Date(document.uploaded_at).toLocaleDateString(locale)}
                    </span>
                  </div>
                </label>
                <button
                  type="button"
                  className="document-delete"
                  aria-label={t('confirmDelete', { name: document.original_name })}
                  onClick={() => handleDelete(document.id, document.original_name)}
                  disabled={busy}
                >
                  x
                </button>
              </div>
            ))
          ) : (
            <div className="empty-panel">{t('noDocuments')}</div>
          )}
        </div>

        <div className="corpus-card surface-card spaced-card">
          <span className="status-label">{t('activeCorpus')}</span>
          <h3>{activeCorpus?.dataset_label || t('nothingSelected')}</h3>
          <p>{activeCorpus ? t('chunksReady', { count: activeCorpus.chunk_count }) : t('selectCorpusHelp')}</p>
          {selectedDocuments.length ? (
            <p className="corpus-selection-note">
              {selectedReadyCount === selectedDocuments.length
                ? t('selectedDocumentsReady', {
                    count: selectedDocuments.length,
                    chunks: selectedCachedChunkCount,
                  })
                : t('selectedDocumentsPartialReady', {
                    ready: selectedReadyCount,
                    total: selectedDocuments.length,
                  })}
            </p>
          ) : null}
          {activationState ? (
            <TaskStatusCard
              eyebrow={t('activeCorpus')}
              title={activationState.title}
              detail={activationState.detail}
              progress={activationState.progress}
            />
          ) : null}
          <button
            type="button"
            className="action-button"
            onClick={handleActivate}
            disabled={!sessionId || !hasSelection || busy}
          >
            {t('activateSelectedFiles')}
          </button>
        </div>
      </div>
    </section>
  );
}

function activationIntroDetail(state, t) {
  if (state.allCached) {
    return t('selectedDocumentsReady', {
      count: state.documentCount,
      chunks: state.cachedChunkCount,
    });
  }
  return t('selectedDocumentsPartialReady', {
    ready: state.readyCount,
    total: state.documentCount,
  });
}

function activationStage(elapsed, state, t) {
  if (state.allCached) {
    if (elapsed < 1.2) {
      return {
        progress: 22,
        title: t('preparingCorpus'),
        detail: activationIntroDetail(state, t),
      };
    }
    if (elapsed < 3.2) {
      return {
        progress: 58,
        title: t('loadingCachedChunks'),
        detail: t('loadingCachedChunksHelp', { count: state.documentCount }),
      };
    }
    if (elapsed < 7.5) {
      return {
        progress: 88,
        title: t('buildingCombinedIndex'),
        detail: t('buildingCombinedIndexHelp'),
      };
    }
    return {
      progress: 94,
      title: t('finalizingCorpus'),
      detail: t('finalizingCorpusHelp'),
    };
  }

  if (elapsed < 1.2) {
    return {
      progress: 16,
      title: t('preparingCorpus'),
      detail: activationIntroDetail(state, t),
    };
  }
  if (elapsed < 3.6) {
    return {
      progress: 42,
      title: t('chunkingSelectedDocuments'),
      detail: t('chunkingSelectedDocumentsHelp'),
    };
  }
  if (elapsed < 8.5) {
    return {
      progress: 80,
      title: t('buildingCombinedIndex'),
      detail: t('buildingCombinedIndexHelp'),
    };
  }
  return {
    progress: 94,
    title: t('finalizingCorpus'),
    detail: t('finalizingCorpusHelp'),
  };
}
