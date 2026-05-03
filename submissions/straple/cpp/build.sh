#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PYBIND_INCLUDE="$($HOME/.local/bin/uv run python -c 'import pybind11; print(pybind11.get_include())')"
PY_INCLUDE="$($HOME/.local/bin/uv run python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_EXT="$($HOME/.local/bin/uv run python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

OUTPUT="$SCRIPT_DIR/_placer_core${PY_EXT}"

c++ -O3 -Wall -Wextra -std=c++17 -shared -fPIC -undefined dynamic_lookup \
    -I"$PYBIND_INCLUDE" -I"$PY_INCLUDE" \
    "$SCRIPT_DIR/placer_core.cpp" -o "$OUTPUT"

echo "Built $OUTPUT"
