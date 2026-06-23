#!/usr/bin/env node
/**
 * Boot helper for a local llama.cpp server.
 *
 * On `npm run dev`, this script:
 *   1. Loads `.env` so it sees LLAMA_CPP_* and LLAMA_SERVER_* variables.
 *   2. Probes the configured base URL.
 *   3. If nothing is listening, spawns `llama-server` with reasonable
 *      Jetson-tuned defaults and the configured Gemma model.
 *   4. Waits until the OpenAI-compatible /v1/models endpoint responds before
 *      letting the rest of the dev stack continue starting.
 *
 * If llama-server cannot be located the script prints a clear hint and exits
 * with a non-zero status so `npm run dev` fails loudly.
 */

const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { spawn, spawnSync } = require('node:child_process');

function loadProjectEnv() {
  const envPath = path.join(__dirname, '..', '.env');
  if (!fs.existsSync(envPath)) return;
  const contents = fs.readFileSync(envPath, 'utf8');
  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const separator = line.indexOf('=');
    if (separator <= 0) continue;
    const key = line.slice(0, separator).trim();
    if (!key || process.env[key] != null) continue;
    let value = line.slice(separator + 1).trim();
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))
    ) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

loadProjectEnv();

const baseUrl = (process.env.LLAMA_CPP_BASE_URL || process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11436').replace(/\/$/, '');
const parsed = new URL(baseUrl);
const host = parsed.hostname || '127.0.0.1';
const port = Number(parsed.port || '11436');

function fetchJson(pathname, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const req = http.get(
      { hostname: host, port, path: pathname, timeout: timeoutMs },
      (res) => {
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          body += chunk;
        });
        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              resolve(JSON.parse(body));
            } catch (error) {
              reject(new Error(`llama-server returned invalid JSON from ${pathname}.`));
            }
            return;
          }
          reject(new Error(`llama-server probe failed with status ${res.statusCode || 'unknown'}.`));
        });
      },
    );
    req.on('timeout', () => req.destroy(new Error('llama-server probe timed out.')));
    req.on('error', reject);
  });
}

async function waitForReady(timeoutMs = 90_000) {
  const startedAt = Date.now();
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      await fetchJson('/v1/models');
      return;
    } catch (error) {
      if (Date.now() - startedAt >= timeoutMs) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 750));
    }
  }
}

function extractModelIds(payload) {
  const models = Array.isArray(payload?.data) ? payload.data : [];
  return models
    .map((item) => (item && typeof item === 'object' ? String(item.id || '').trim() : ''))
    .filter(Boolean);
}

function resolveLlamaServerBinary() {
  const explicit = process.env.LLAMA_SERVER_BIN;
  if (explicit && fs.existsSync(explicit)) {
    return explicit;
  }
  const candidates = [
    process.env.HOME ? path.join(process.env.HOME, 'llama.cpp', 'build', 'bin', 'llama-server') : null,
    process.env.HOME ? path.join(process.env.HOME, 'llama.cpp', 'llama-server') : null,
    '/usr/local/bin/llama-server',
    '/usr/bin/llama-server',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  const which = spawnSync('which', ['llama-server']);
  if (which.status === 0) {
    return which.stdout.toString().trim();
  }
  return '';
}

function resolveModelPath() {
  const explicit = process.env.LLAMA_CPP_MODEL_PATH;
  if (explicit) return explicit;
  const candidates = [
    process.env.HOME ? path.join(process.env.HOME, 'models', 'gemma4', 'gemma-4-E2B-it-Q4_K_M.gguf') : null,
    process.env.HOME ? path.join(process.env.HOME, 'models', 'gemma-4-E2B-it-Q4_K_M.gguf') : null,
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return '';
}

function resolveHfRepo() {
  const explicit = (process.env.LLAMA_CPP_HF_REPO || '').trim();
  if (explicit) return explicit;
  return '';
}

function resolveMmprojPath() {
  if (String(process.env.LLAMA_CPP_VISION_ENABLED || '1').trim().toLowerCase() === '0') {
    return '';
  }
  const explicit = process.env.LLAMA_CPP_MMPROJ_PATH;
  if (explicit) {
    if (fs.existsSync(explicit)) return explicit;
    console.warn(`[llama.cpp] LLAMA_CPP_MMPROJ_PATH points to a missing file: ${explicit}`);
    console.warn('            Vision (image input) will be disabled.');
    return '';
  }
  const candidates = [
    process.env.HOME ? path.join(process.env.HOME, 'models', 'gemma4', 'mmproj-F16.gguf') : null,
    process.env.HOME ? path.join(process.env.HOME, 'models', 'mmproj-F16.gguf') : null,
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return '';
}

function buildArgs({ modelPath, hfRepo, mmprojPath }) {
  const args = [
    '--host', host,
    '--port', String(port),
    '-c', process.env.LLAMA_CPP_NUM_CTX || process.env.EDGE_RAG_NUM_CTX || '6144',
    '-np', process.env.LLAMA_CPP_PARALLEL || '1',
    '-ngl', process.env.LLAMA_CPP_NGL || '99',
    '--alias', process.env.LLAMA_CPP_MODEL || process.env.EDGE_RAG_ANSWER_MODEL || 'gemma-4-e2b-q4km',
  ];
  if (hfRepo) {
    args.unshift(hfRepo);
    args.unshift('-hf');
    const hfFile = (process.env.LLAMA_CPP_HF_FILE || '').trim();
    if (hfFile) {
      args.push('--hf-file', hfFile);
    }
  } else {
    args.unshift(modelPath);
    args.unshift('-m');
  }
  if (process.env.LLAMA_CPP_THREADS) {
    args.push('-t', process.env.LLAMA_CPP_THREADS);
  }
  const flashAttn = String(process.env.LLAMA_CPP_FLASH_ATTN || '1').trim().toLowerCase();
  if (flashAttn && flashAttn !== '0' && flashAttn !== 'off') {
    args.push('-fa', flashAttn === '1' || flashAttn === 'true' ? 'on' : flashAttn);
  }
  // Expose embeddings on the same port so RAG can reuse the loaded model.
  // llama-server's OpenAI-compatible /v1/embeddings rejects pooling=none, so
  // default to a safe OAI-compatible mode unless the environment overrides it.
  if ((process.env.LLAMA_CPP_EMBEDDINGS || '1') !== '0') {
    args.push('--embeddings');
    const pooling = String(process.env.LLAMA_CPP_POOLING || 'mean').trim().toLowerCase();
    if (pooling && pooling !== 'default') {
      args.push('--pooling', pooling);
    }
  }
  // Hook up the multimodal projector so the model can read images. On unified
  // memory boards like Jetson, keeping the projector on the CPU avoids the
  // "failed to allocate compute pp buffers" OOM when the GGUF model already
  // fills the GPU; flip LLAMA_CPP_MMPROJ_OFFLOAD=1 to put it back on GPU.
  if (mmprojPath) {
    args.push('--mmproj', mmprojPath);
    if ((process.env.LLAMA_CPP_MMPROJ_OFFLOAD || '0') !== '1') {
      args.push('--no-mmproj-offload');
    }
    // Quantize the KV cache when vision is enabled - image tokens add a lot of
    // additional context, so we trade a tiny bit of quality for ~50% less KV
    // memory and avoid CUDA "failed to allocate compute pp buffers" crashes
    // during image decoding on memory-constrained boards.
    const kvType = process.env.LLAMA_CPP_KV_CACHE_TYPE || 'q8_0';
    if (kvType && kvType.toLowerCase() !== 'off') {
      args.push('--cache-type-k', kvType);
      args.push('--cache-type-v', kvType);
    }
  }
  // Route thinking/analysis tokens into `message.reasoning_content` (and the
  // matching `delta.reasoning_content` chunks) so the UI can render them in a
  // separate "Analyzing..." panel instead of mixing them with the final answer.
  const reasoningMode = String(process.env.LLAMA_CPP_REASONING || 'auto').trim().toLowerCase();
  if (reasoningMode && reasoningMode !== '0' && reasoningMode !== 'off') {
    args.push('--reasoning-format', 'deepseek');
    args.push('-rea', reasoningMode === '1' || reasoningMode === 'true' ? 'on' : reasoningMode);
    if (process.env.LLAMA_CPP_REASONING_BUDGET) {
      args.push('--reasoning-budget', String(process.env.LLAMA_CPP_REASONING_BUDGET));
    }
  }
  if (process.env.LLAMA_CPP_EXTRA_ARGS) {
    args.push(...process.env.LLAMA_CPP_EXTRA_ARGS.split(/\s+/).filter(Boolean));
  }
  return args;
}

async function main() {
  const configuredModel = (process.env.LLAMA_CPP_MODEL || process.env.EDGE_RAG_ANSWER_MODEL || 'gemma-4-e2b-q4km').trim();
  try {
    const payload = await fetchJson('/v1/models');
    const loadedModels = extractModelIds(payload);
    if (!configuredModel || loadedModels.includes(configuredModel)) {
      console.log(`[llama.cpp] Using existing server at ${baseUrl}.`);
      setInterval(() => {}, 1 << 30);
      return;
    }
    console.error(`[llama.cpp] A server is already running at ${baseUrl}, but it does not have the configured model loaded.`);
    console.error(`[llama.cpp] Expected: ${configuredModel}`);
    console.error(`[llama.cpp] Loaded: ${loadedModels.join(', ') || '(none reported)'}`);
    console.error('[llama.cpp] Stop the existing llama-server and run `npm run dev` again.');
    process.exit(1);
  } catch {
    // Server isn't up yet — try to launch it.
  }

  const binary = resolveLlamaServerBinary();
  if (!binary) {
    console.error('[llama.cpp] Could not find a llama-server binary.');
    console.error('            Set LLAMA_SERVER_BIN to its path or add it to your PATH.');
    console.error('            Expected location: ~/llama.cpp/build/bin/llama-server');
    process.exit(1);
  }
  const hfRepo = resolveHfRepo();
  const modelPath = hfRepo ? '' : resolveModelPath();
  if (!hfRepo && !modelPath) {
    console.error('[llama.cpp] Could not find a GGUF model to load.');
    console.error('            Set LLAMA_CPP_MODEL_PATH to its absolute path,');
    console.error('            or set LLAMA_CPP_HF_REPO to a Hugging Face GGUF repo spec,');
    console.error('            e.g. ibm-research/granite-vision-3.2-2b-GGUF:Q4_K_M');
    process.exit(1);
  }

  const mmprojPath = hfRepo ? '' : resolveMmprojPath();
  if (hfRepo) {
    console.log(`[llama.cpp] Using Hugging Face GGUF repo: ${hfRepo}`);
    console.log('[llama.cpp] Any matching multimodal projector will be fetched automatically by llama.cpp.');
  } else if (mmprojPath) {
    console.log(`[llama.cpp] Vision support enabled via mmproj: ${mmprojPath}`);
  } else {
    console.log('[llama.cpp] No mmproj found - image input will be unavailable.');
  }
  const args = buildArgs({ modelPath, hfRepo, mmprojPath });
  console.log(`[llama.cpp] Starting ${binary} ${args.join(' ')}`);
  const child = spawn(binary, args, {
    stdio: 'inherit',
    env: { ...process.env },
  });
  const forwardSignal = (signal) => {
    if (!child.killed) child.kill(signal);
  };
  process.on('SIGINT', () => forwardSignal('SIGINT'));
  process.on('SIGTERM', () => forwardSignal('SIGTERM'));
  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });

  try {
    await waitForReady();
    console.log(`[llama.cpp] Server is ready at ${baseUrl}.`);
  } catch (error) {
    console.error(`[llama.cpp] Server did not become ready in time: ${error.message}`);
    forwardSignal('SIGTERM');
    process.exit(1);
  }

  setInterval(() => {}, 1 << 30);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
