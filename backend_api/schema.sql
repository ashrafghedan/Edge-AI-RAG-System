CREATE TABLE users (
    id VARCHAR(64) PRIMARY KEY,
    email VARCHAR(320) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    password_hash TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE auth_tokens (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(128) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE app_sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    is_library BOOLEAN NOT NULL DEFAULT FALSE,
    active_corpus_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE documents (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES app_sessions(id) ON DELETE CASCADE,
    user_id VARCHAR(64) REFERENCES users(id) ON DELETE CASCADE,
    original_name VARCHAR(255) NOT NULL,
    storage_path TEXT NOT NULL,
    content_text TEXT NOT NULL DEFAULT '',
    sha256 VARCHAR(128) NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at TIMESTAMPTZ NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE corpora (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES app_sessions(id) ON DELETE CASCADE,
    dataset_id VARCHAR(128) NOT NULL,
    dataset_label VARCHAR(255) NOT NULL,
    vector_directory TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    source_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE app_sessions
    ADD CONSTRAINT fk_app_sessions_active_corpus
    FOREIGN KEY (active_corpus_id) REFERENCES corpora(id) ON DELETE SET NULL;

CREATE TABLE corpus_documents (
    corpus_id VARCHAR(64) NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    document_id VARCHAR(64) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    PRIMARY KEY (corpus_id, document_id)
);

CREATE TABLE chat_messages (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES app_sessions(id) ON DELETE CASCADE,
    mode VARCHAR(32) NOT NULL,
    role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE generated_questions (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES app_sessions(id) ON DELETE CASCADE,
    corpus_id VARCHAR(64) REFERENCES corpora(id) ON DELETE SET NULL,
    question_text TEXT NOT NULL,
    model_answer TEXT NOT NULL,
    source_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE answer_attempts (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES app_sessions(id) ON DELETE CASCADE,
    generated_question_id VARCHAR(64) NOT NULL REFERENCES generated_questions(id) ON DELETE CASCADE,
    user_answer TEXT NOT NULL,
    score INTEGER NOT NULL,
    feedback TEXT NOT NULL,
    model_answer TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE session_events (
    id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES app_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_auth_tokens_user_id ON auth_tokens(user_id);
CREATE INDEX idx_app_sessions_user_id ON app_sessions(user_id);
CREATE INDEX idx_documents_session_id ON documents(session_id);
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_corpora_session_id ON corpora(session_id);
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_generated_questions_session_id ON generated_questions(session_id);
CREATE INDEX idx_answer_attempts_session_id ON answer_attempts(session_id);
CREATE INDEX idx_session_events_session_id ON session_events(session_id);
