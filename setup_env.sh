#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Creating virtual environment in $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Activating virtual environment ..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip ..."
pip install --upgrade pip

echo "Installing ksmm in editable mode ..."
pip install -e "$SCRIPT_DIR"

echo ""
echo "Done! To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
