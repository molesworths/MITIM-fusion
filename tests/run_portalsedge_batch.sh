#!/usr/bin/env bash

set -u

WORKFLOW_DIR="${1:-.}"
PATTERN="${2:-*.py}"
PYTHON_BIN="${PYTHON_BIN:-}"
HALT_ON_ERROR="${HALT_ON_ERROR:-0}"
USE_PIXI="${USE_PIXI:-1}"
AUTO_YES="${AUTO_YES:-1}"
AUTO_YES_COUNT="${AUTO_YES_COUNT:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMARTS_ROOT="${SMARTS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ ! -d "$WORKFLOW_DIR" ]]; then
  echo "Error: directory not found: $WORKFLOW_DIR" >&2
  exit 1
fi

if [[ "$USE_PIXI" == "1" ]]; then
  if ! command -v pixi >/dev/null 2>&1; then
    echo "Error: pixi executable not found on PATH." >&2
    exit 1
  fi
else
  if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
      PYTHON_BIN="python"
    else
      echo "Error: no Python interpreter found on PATH (tried python3, python)." >&2
      exit 1
    fi
  elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: python executable not found: $PYTHON_BIN" >&2
    exit 1
  fi
fi

if [[ "$AUTO_YES" == "1" ]] && ! command -v python3 >/dev/null 2>&1; then
  echo "Error: AUTO_YES=1 requires python3 for PTY prompt handling." >&2
  exit 1
fi

get_folder_work() {
  python3 - "$1" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
tree = ast.parse(path.read_text())

for node in tree.body:
  if isinstance(node, ast.Assign):
    for target in node.targets:
      if isinstance(target, ast.Name) and target.id == "folderWork":
        try:
          value = ast.literal_eval(node.value)
        except Exception:
          value = None

        if value is None and isinstance(node.value, ast.Call):
          func = node.value.func
          func_name = None
          if isinstance(func, ast.Name):
            func_name = func.id
          elif isinstance(func, ast.Attribute):
            func_name = func.attr

          if func_name == "Path" and len(node.value.args) == 1:
            try:
              value = ast.literal_eval(node.value.args[0])
            except Exception:
              value = None

        if value is None:
          sys.exit(1)

        print(value)
        sys.exit(0)

sys.exit(1)
PY
}

copy_workflow_to_folder_work() {
  local workflow_path="$1"
  local folder_work
  folder_work="$(get_folder_work "$workflow_path")" || return 1

  if [[ -z "$folder_work" ]]; then
    echo "Error: could not determine folderWork in $workflow_path" >&2
    return 1
  fi

  mkdir -p "$folder_work"
  cp "$workflow_path" "$folder_work/$(basename "$workflow_path")"
}

run_with_auto_yes() {
  python3 - "$@" <<'PY'
import os
import sys
import selectors
import subprocess

cmd = sys.argv[1:]
if not cmd:
  sys.exit(2)

max_replies = int(os.environ.get("AUTO_YES_COUNT", "10"))
replies = 0

master_fd, slave_fd = os.openpty()
proc = subprocess.Popen(
  cmd,
  stdin=slave_fd,
  stdout=slave_fd,
  stderr=slave_fd,
  close_fds=True,
)
os.close(slave_fd)

sel = selectors.DefaultSelector()
sel.register(master_fd, selectors.EVENT_READ)

tail = ""

try:
  while True:
    events = sel.select(timeout=0.1)
    if events:
      try:
        chunk = os.read(master_fd, 4096)
      except OSError:
        break

      if not chunk:
        break

      sys.stdout.buffer.write(chunk)
      sys.stdout.buffer.flush()

      text = chunk.decode("utf-8", errors="ignore")
      tail = (tail + text)[-4096:]

      if ("y/n/e" in tail.lower()) and (replies < max_replies):
        os.write(master_fd, b"y\n")
        replies += 1
        tail = ""

    if proc.poll() is not None:
      # Drain remaining output after process exits.
      while True:
        try:
          chunk = os.read(master_fd, 4096)
        except OSError:
          chunk = b""
        if not chunk:
          break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
      break
finally:
  try:
    sel.unregister(master_fd)
  except Exception:
    pass
  os.close(master_fd)

sys.exit(proc.wait())
PY
}

mapfile -t workflow_files < <(find "$WORKFLOW_DIR" -maxdepth 1 -type f -name "$PATTERN" | sort)

if [[ ${#workflow_files[@]} -eq 0 ]]; then
  echo "No files matched pattern '$PATTERN' in $WORKFLOW_DIR"
  exit 0
fi

echo "Running ${#workflow_files[@]} workflow file(s) from: $WORKFLOW_DIR"
echo "Pattern: $PATTERN"
if [[ "$USE_PIXI" == "1" ]]; then
  echo "Runner : pixi run python"
  echo "Root   : $SMARTS_ROOT"
else
  echo "Python : $PYTHON_BIN"
fi
echo "Auto Y : $AUTO_YES"
if [[ "$AUTO_YES" == "1" ]]; then
  echo "Auto Y count: $AUTO_YES_COUNT"
fi
echo

passed=0
failed=0

for file in "${workflow_files[@]}"; do
  base="$(basename "$file")"

  # Skip this launcher script and common non-workflow modules.
  if [[ "$base" == "run_portalsedge_batch.sh" || "$base" == "__init__.py" ]]; then
    continue
  fi

  file_abs="$(realpath "$file")"

  if ! copy_workflow_to_folder_work "$file_abs"; then
    echo "FAIL : $file (could not copy into folderWork)"
    failed=$((failed + 1))
    if [[ "$HALT_ON_ERROR" == "1" ]]; then
      echo "Stopping after first failure (HALT_ON_ERROR=1)."
      exit 1
    fi
    echo
    continue
  fi

  echo "=== Running: $file"
  if [[ "$USE_PIXI" == "1" ]]; then
    (
      cd "$SMARTS_ROOT" || exit 1
      if [[ "$AUTO_YES" == "1" ]]; then
        run_with_auto_yes pixi run python "$file_abs"
      else
        pixi run python "$file_abs"
      fi
    )
    status=$?
  else
    if [[ "$AUTO_YES" == "1" ]]; then
      run_with_auto_yes "$PYTHON_BIN" "$file_abs"
      status=$?
    else
      "$PYTHON_BIN" "$file_abs"
      status=$?
    fi
  fi

  if [[ $status -eq 0 ]]; then
    ((passed++))
    echo "OK   : $file"
  else
    ((failed++))
    echo "FAIL : $file (exit $status)"
    if [[ "$HALT_ON_ERROR" == "1" ]]; then
      echo "Stopping after first failure (HALT_ON_ERROR=1)."
      exit $status
    fi
  fi

  echo

done

echo "Done. Passed: $passed  Failed: $failed"
if [[ $failed -gt 0 ]]; then
  exit 1
fi
