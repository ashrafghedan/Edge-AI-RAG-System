import { useEffect, useMemo, useRef, useState } from 'react';

import { transcribeChatAudio } from '../../api/chat';
import { usePreferences } from '../../preferences';

function getRecognitionConstructor() {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function canRecordAudio() {
  return (
    typeof window !== 'undefined'
    && typeof window.MediaRecorder !== 'undefined'
    && typeof navigator !== 'undefined'
    && !!navigator.mediaDevices?.getUserMedia
  );
}

function chooseRecordingMimeType() {
  if (typeof window === 'undefined' || typeof window.MediaRecorder?.isTypeSupported !== 'function') {
    return '';
  }
  for (const candidate of ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg']) {
    if (window.MediaRecorder.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return '';
}

function mergeTranscript(baseText, transcript) {
  const trimmedBase = baseText.trimEnd();
  const trimmedTranscript = transcript.trim();
  if (!trimmedTranscript) {
    return trimmedBase;
  }
  if (!trimmedBase) {
    return trimmedTranscript;
  }
  return `${trimmedBase} ${trimmedTranscript}`;
}

function normalizeSpeechLanguage(language) {
  const lowered = String(language || '').toLowerCase();
  if (lowered.startsWith('ar')) return 'ar';
  if (lowered.startsWith('en')) return 'en';
  return lowered || 'en';
}

export default function SpeechToTextButton({
  sessionId,
  value,
  onValueChange,
  onError,
  disabled = false,
  className = '',
  idleLabel,
  listeningLabel,
}) {
  const { speechLanguage, t } = usePreferences();
  const recognitionRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const baseTextRef = useRef(value || '');
  const [listening, setListening] = useState(false);
  const [transcribing, setTranscribing] = useState(false);

  const RecognitionConstructor = useMemo(() => getRecognitionConstructor(), []);
  const supportsBrowserRecognition = Boolean(RecognitionConstructor);
  const supportsBackendRecording = canRecordAudio() && Boolean(sessionId);
  const supported = supportsBackendRecording || supportsBrowserRecognition;

  useEffect(() => {
    if (!listening && !transcribing) {
      baseTextRef.current = value || '';
    }
  }, [value, listening, transcribing]);

  useEffect(
    () => () => {
      recognitionRef.current?.stop?.();
      recorderRef.current?.stop?.();
      stopStream(streamRef.current);
      recognitionRef.current = null;
      recorderRef.current = null;
      streamRef.current = null;
    },
    [],
  );

  const reportError = (message) => {
    if (typeof onError === 'function') {
      onError(message);
    }
  };

  const handleClick = async () => {
    if (!supported || disabled || transcribing) {
      return;
    }

    if (listening) {
      recognitionRef.current?.stop?.();
      recorderRef.current?.stop?.();
      return;
    }

    reportError('');
    if (supportsBackendRecording) {
      await startRecordedTranscription({
        sessionId,
        value,
        speechLanguage,
        onValueChange,
        onError: reportError,
        baseTextRef,
        recorderRef,
        streamRef,
        chunksRef,
        setListening,
        setTranscribing,
      });
      return;
    }

    startBrowserRecognition({
      RecognitionConstructor,
      value,
      speechLanguage,
      onValueChange,
      onError: reportError,
      baseTextRef,
      recognitionRef,
      setListening,
    });
  };

  const title = !supported
    ? t('speechUnsupported')
    : transcribing
      ? t('pleaseWait')
      : listening
        ? t('stopVoiceInput')
        : t('startVoiceInput');

  return (
    <button
      type="button"
      className={`speech-button ${listening ? 'speech-button-listening' : ''} ${className}`.trim()}
      onClick={handleClick}
      disabled={disabled || !supported || transcribing}
      title={title}
      aria-pressed={listening}
      aria-label={listening ? listeningLabel || t('stopVoiceInput') : idleLabel || title}
    >
      <svg className="speech-icon" viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M12 15a3 3 0 0 0 3-3V7a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm5-3a1 1 0 1 1 2 0 7 7 0 0 1-6 6.92V21h3a1 1 0 1 1 0 2H8a1 1 0 1 1 0-2h3v-2.08A7 7 0 0 1 5 12a1 1 0 0 1 2 0 5 5 0 0 0 10 0Z"
          fill="currentColor"
        />
      </svg>
    </button>
  );
}

async function startRecordedTranscription({
  sessionId,
  value,
  speechLanguage,
  onValueChange,
  onError,
  baseTextRef,
  recorderRef,
  streamRef,
  chunksRef,
  setListening,
  setTranscribing,
}) {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (_error) {
    onError('Microphone access is unavailable. Check browser permissions and audio devices.');
    return;
  }

  const mimeType = chooseRecordingMimeType();
  try {
    const recorder = mimeType ? new window.MediaRecorder(stream, { mimeType }) : new window.MediaRecorder(stream);
    baseTextRef.current = value || '';
    chunksRef.current = [];
    streamRef.current = stream;
    recorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data?.size) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onerror = () => {
      setListening(false);
      setTranscribing(false);
      onError('Audio recording failed before transcription could start.');
      stopStream(streamRef.current);
      streamRef.current = null;
      recorderRef.current = null;
    };

    recorder.onstop = async () => {
      setListening(false);
      stopStream(streamRef.current);
      streamRef.current = null;
      recorderRef.current = null;

      if (!chunksRef.current.length) {
        return;
      }

      setTranscribing(true);
      try {
        const extension = mimeType.includes('ogg') ? 'ogg' : mimeType.includes('wav') ? 'wav' : 'webm';
        const audioBlob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' });
        const audioFile = new File([audioBlob], `speech-${Date.now()}.${extension}`, {
          type: mimeType || 'audio/webm',
        });
        const response = await transcribeChatAudio(sessionId, audioFile, {
          language: normalizeSpeechLanguage(speechLanguage),
        });
        onValueChange(mergeTranscript(baseTextRef.current, response.text || ''));
      } catch (error) {
        onError(error?.message || 'Speech transcription failed.');
      } finally {
        chunksRef.current = [];
        setTranscribing(false);
      }
    };

    setListening(true);
    recorder.start();
  } catch (_error) {
    stopStream(stream);
    onError('Audio recording is not supported in this browser environment.');
  }
}

function startBrowserRecognition({
  RecognitionConstructor,
  value,
  speechLanguage,
  onValueChange,
  onError,
  baseTextRef,
  recognitionRef,
  setListening,
}) {
  const recognition = new RecognitionConstructor();
  baseTextRef.current = value || '';
  recognition.lang = speechLanguage;
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map((result) => result[0]?.transcript || '')
      .join(' ');
    onValueChange(mergeTranscript(baseTextRef.current, transcript));
  };

  recognition.onerror = () => {
    setListening(false);
    recognitionRef.current = null;
    onError('Voice recognition failed in the browser.');
  };

  recognition.onend = () => {
    setListening(false);
    recognitionRef.current = null;
  };

  recognitionRef.current = recognition;
  setListening(true);
  recognition.start();
}

function stopStream(stream) {
  stream?.getTracks?.().forEach((track) => track.stop());
}
