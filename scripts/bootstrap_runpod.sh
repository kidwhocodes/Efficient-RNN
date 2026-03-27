#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  apt-get update
  apt-get install -y tmux
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e ./neurogym
pip install -e ./Mod_Cog
pip install -e .

mkdir -p /tmp/mpl
echo "Bootstrap complete. Next run:"
echo "  MPLCONFIGDIR=/tmp/mpl python scripts/runpod_preflight.py"
