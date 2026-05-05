#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PYBIND_INCLUDE="$($HOME/.local/bin/uv run python -c 'import pybind11; print(pybind11.get_include())')"
PY_INCLUDE="$($HOME/.local/bin/uv run python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_EXT="$($HOME/.local/bin/uv run python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

PLACER_OUT="$SCRIPT_DIR/_placer_core${PY_EXT}"
PROXY_OUT="$SCRIPT_DIR/_proxy_cost${PY_EXT}"

UNDEF_FLAG=""
if [[ "$(uname -s)" == "Darwin" ]]; then
    UNDEF_FLAG="-undefined dynamic_lookup"
fi

c++ -O3 -Wall -Wextra -std=c++17 -shared -fPIC $UNDEF_FLAG \
    -I"$PYBIND_INCLUDE" -I"$PY_INCLUDE" \
    "$SCRIPT_DIR/placer_core.cpp" -o "$PLACER_OUT"

c++ -O3 -Wall -Wextra -std=c++17 -shared -fPIC $UNDEF_FLAG \
    -I"$PYBIND_INCLUDE" -I"$PY_INCLUDE" \
    "$SCRIPT_DIR/proxy_cost.cpp" -o "$PROXY_OUT"

echo "Built $PLACER_OUT"
echo "Built $PROXY_OUT"
