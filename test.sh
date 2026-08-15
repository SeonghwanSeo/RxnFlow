#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"
case "$mode" in
  quick|heavy|all) ;;
  *) echo "usage: $0 {quick|heavy|all}" >&2; exit 2 ;;
esac

if [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin=python3.10
fi

quick() {
  "$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
  "$python_bin" -m compileall -q src tests
  ruff check src tests
  "$python_bin" -m build --no-isolation
  PYTHONPATH=src "$python_bin" -c 'import rxnflow; print(rxnflow.__all__)'
  PYTHONPATH=src "$python_bin" -m pytest -m 'not heavy' tests
}

heavy() {
  PYTHONPATH=src "$python_bin" -m pytest -m heavy tests/test_heavy.py
}

if [[ "$mode" == quick || "$mode" == all ]]; then quick; fi
if [[ "$mode" == heavy || "$mode" == all ]]; then heavy; fi
