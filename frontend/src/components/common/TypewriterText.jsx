import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function typingStepForLength(length) {
  // Reveal a few characters at a time so the typewriter feels smooth rather
  // than mechanical. Longer outputs use bigger chunks so we never lag behind.
  if (length <= 240) return 2;
  if (length <= 720) return 3;
  if (length <= 1400) return 5;
  return Math.max(6, Math.ceil(length / 320));
}

function delayForCharacter(character, index) {
  if (character === '\n') return 70;
  if (/[.!?]/.test(character)) return 80;
  if (/[,;:]/.test(character)) return 48;
  if (character === ' ') return 20;
  return 14 + ((index % 5) * 2);
}

function codeLanguageFromClassName(className = '') {
  const match = /language-([\w-]+)/i.exec(className);
  return match?.[1]?.toLowerCase() || 'text';
}

function codeText(children) {
  if (Array.isArray(children)) {
    return children.join('');
  }
  return String(children ?? '');
}

const markdownComponents = {
  a({ href, children, ...props }) {
    return (
      <a href={href} target="_blank" rel="noreferrer noopener" {...props}>
        {children}
      </a>
    );
  },
  pre({ children }) {
    return children;
  },
  code({ inline, className, children, ...props }) {
    const content = codeText(children).replace(/\n$/, '');
    if (inline) {
      return (
        <code className={className} {...props}>
          {content}
        </code>
      );
    }

    return (
      <div className="markdown-code-block">
        <div className="markdown-code-header" aria-hidden="true">
          <span className="markdown-code-dots">
            <span className="markdown-code-dot" />
            <span className="markdown-code-dot" />
            <span className="markdown-code-dot" />
          </span>
          <span className="markdown-code-label">{codeLanguageFromClassName(className)}</span>
        </div>
        <pre>
          <code className={className} {...props}>
            {content}
          </code>
        </pre>
      </div>
    );
  },
  table({ children }) {
    return (
      <div className="markdown-table-shell">
        <table>{children}</table>
      </div>
    );
  },
};

function MarkdownBlock({ text, className, isAnimating = false }) {
  return (
    <div className={`markdown-block ${className}`.trim()}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {text}
      </ReactMarkdown>
      {isAnimating ? <span className="typewriter-caret markdown-typewriter-caret" aria-hidden="true" /> : null}
    </div>
  );
}

export default function TypewriterText({
  text,
  animate = false,
  as = 'p',
  className = '',
  onDone,
  onProgress,
  markdown = false,
  streaming = false,
}) {
  const effectiveAnimate = animate && !streaming;
  const [visibleText, setVisibleText] = useState(() => (effectiveAnimate ? '' : text));
  const onDoneRef = useRef(onDone);
  const onProgressRef = useRef(onProgress);

  useEffect(() => {
    onDoneRef.current = onDone;
  }, [onDone]);

  useEffect(() => {
    onProgressRef.current = onProgress;
  }, [onProgress]);

  useEffect(() => {
    if (!effectiveAnimate || !text) {
      setVisibleText(text);
      if (effectiveAnimate) {
        onProgressRef.current?.();
        onDoneRef.current?.();
      }
      return undefined;
    }

    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setVisibleText(text);
      onProgressRef.current?.();
      onDoneRef.current?.();
      return undefined;
    }

    let index = 0;
    let timer = 0;
    const step = typingStepForLength(text.length);
    setVisibleText('');

    const revealNextChunk = () => {
      index = Math.min(text.length, index + step);
      setVisibleText(text.slice(0, index));
      onProgressRef.current?.();

      if (index >= text.length) {
        onDoneRef.current?.();
        return;
      }

      timer = window.setTimeout(revealNextChunk, delayForCharacter(text[index - 1], index));
    };

    timer = window.setTimeout(revealNextChunk, 80);

    return () => window.clearTimeout(timer);
  }, [effectiveAnimate, text]);

  const Component = as;
  const isAnimating = (effectiveAnimate && visibleText.length < text.length) || streaming;

  if (markdown) {
    return <MarkdownBlock text={visibleText} className={className} isAnimating={isAnimating} />;
  }

  return (
    <Component className={className}>
      {visibleText}
      {isAnimating ? <span className="typewriter-caret" aria-hidden="true" /> : null}
    </Component>
  );
}
