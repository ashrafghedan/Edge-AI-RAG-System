# Edge AI RAG System for Knowledge Delivery and Security

A fully **offline-first**, on-device AI learning assistant. Upload your own study
material (PDF or text), then ask grounded questions whose answers cite the
retrieved evidence, auto-generate reading-comprehension questions, get graded
answers with feedback, and use offline voice input — all powered by a **local**
language model. No cloud, no internet dependency, and no user data ever leaves
the device.

**Graduation Project — The Hashemite University, Faculty of Engineering,
Computer Engineering Department.**

---

## Architecture

| Layer            | Technology                                                        |
|------------------|-------------------------------------------------------------------|
| Language model   | Quantized **Gemma E2B** served by **llama.cpp** (OpenAI-compatible) |
| Vector store     | **ChromaDB** (on-device embeddings, per-corpus index)             |
| Backend          | **FastAPI** + SQLAlchemy (Python)                                  |
| Database         | **PostgreSQL**                                                     |
| Frontend         | **React** + Vite                                                   |
| Document pipeline| PyMuPDF + Tesseract OCR fallback, overlapping chunking            |
| Speech-to-text   | **faster-whisper** (offline)                                      |
| Security         | PBKDF2 password hashing, hashed bearer tokens, local safety policy |

`npm run dev` launches three processes together: the **llama.cpp** model server,
the **FastAPI** backend, and the **Vite** frontend.

---

## Prerequisites

Install these first:

- **Node.js** 18+ and npm
- **Python** 3.12
- **PostgreSQL** 14+ (running locally)
- **llama.cpp** — the `llama-server` binary
  (download a prebuilt release or build from source; a CUDA build is recommended
  if you have an NVIDIA GPU, otherwise a CPU build works)
- A **GGUF model** file — e.g. `gemma-3n-E2B-it-Q4_K_M.gguf`
- *(Optional)* **Tesseract OCR** for scanned-PDF text extraction

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/ashrafghedan/graduation-project.git
cd graduation-project
```

### 2. Python backend dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux:
# .venv/bin/python -m pip install -r requirements.txt
```

### 3. Node dependencies

```bash
npm install
npm --prefix frontend install
```

### 4. PostgreSQL database

Create an empty database (the schema is created automatically on first run):

```bash
psql -U postgres -c "CREATE DATABASE edge_rag;"
```

### 5. Configure environment

```bash
cp .env.example .env
```

Then edit `.env` and set, at minimum:

- `DATABASE_URL` — your PostgreSQL user/password/host/database
- `LLAMA_SERVER_BIN` — absolute path to your `llama-server` binary
- `LLAMA_CPP_MODEL_PATH` — absolute path to your GGUF model

---

## Run

```bash
npm run dev
```

This starts all three services. Once they're up, open:

> **http://localhost:5173**

The first launch takes ~30–60s while the model loads into memory. Create an
account, upload a PDF or text file, activate it as a corpus, and start asking
grounded questions.

### Services / ports

| Service        | URL                          |
|----------------|------------------------------|
| Frontend (Vite)| http://localhost:5173        |
| Backend (API)  | http://127.0.0.1:8001        |
| Model server   | http://127.0.0.1:11436       |

Health check: `http://127.0.0.1:8001/api/v1/health`

---

## Features

- **Grounded Q&A** — answers cite retrieved source chunks and decline to answer
  when the documents don't support the question.
- **Document ingestion** — PDF/text upload with OCR fallback, segmented into
  overlapping chunks and embedded on-device.
- **Auto question generation** — reading-comprehension questions from your corpus.
- **AI grading** — 0–10 scores with feedback from the same local model.
- **Offline voice input** — transcribed on-device with faster-whisper.
- **Security by design** — PBKDF2 hashing, hashed bearer tokens, local
  content-safety policy, and complete data locality.

---

## Notes

- **Vision (image input):** disabled by default. To enable it, point
  `LLAMA_CPP_MMPROJ_PATH` at a multimodal projector GGUF and set
  `LLAMA_CPP_VISION_ENABLED=1`.
- **No GPU?** Use a CPU build of `llama-server`; lower `LLAMA_CPP_NUM_CTX` and
  `LLAMA_CPP_MAX_TOKENS` for smaller machines.
- The `.env` file, the model files, `node_modules/`, the Python virtual
  environment, and the local `data/` directory are intentionally **not** tracked
  in git.
