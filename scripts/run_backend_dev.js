const { existsSync } = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const projectRoot = path.resolve(__dirname, '..');
const backendRunner = path.join(projectRoot, 'scripts', 'run_backend_dev.py');
const isWindows = process.platform === 'win32';
const candidateCommands = isWindows
  ? [
      [path.join(projectRoot, '.venv', 'Scripts', 'python.exe')],
      ['python'],
      ['python3'],
    ]
  : [
      [path.join(projectRoot, '.venv', 'bin', 'python')],
      [path.join(projectRoot, '.venv', 'bin', 'python3')],
      ['python3'],
      ['python'],
    ];

for (const [command] of candidateCommands) {
  if (path.isAbsolute(command) && !existsSync(command)) {
    continue;
  }

  const result = spawnSync(command, [backendRunner], {
    cwd: path.join(projectRoot, 'backend_api'),
    stdio: 'inherit',
  });

  if (result.error) {
    if (result.error.code === 'ENOENT') {
      continue;
    }
    throw result.error;
  }

  process.exit(result.status === null ? 1 : result.status);
}

console.error('Unable to find a Python interpreter for the backend runner.');
console.error(
  isWindows
    ? 'Expected one of: .venv/Scripts/python.exe, python, or python3.'
    : 'Expected one of: .venv/bin/python, .venv/bin/python3, python3, or python.',
);
process.exit(1);
