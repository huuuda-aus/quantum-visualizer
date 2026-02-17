#!/usr/bin/env bash
set -euo pipefail

# Build telemetry (optional but recommended)
# cargo run -p generator --release

TARGET=wasm32-unknown-unknown
DIST=dist

mkdir -p "$DIST"

# Ensure rustup/cargo are available in CI environments (Netlify may not have them).
if ! command -v cargo >/dev/null 2>&1; then
  echo "rustup/cargo not found — installing rustup (non-interactive)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  export PATH="$HOME/.cargo/bin:$PATH"
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env" || true
fi

# Ensure nightly toolchain is installed for the +nightly cargo invocation used below.
if ! rustup toolchain list | grep -q "^nightly"; then
  echo "Installing Rust nightly toolchain..."
  rustup toolchain install nightly
fi

rustup target add "$TARGET" >/dev/null 2>&1 || true

# Note: the workspace release profile enables LTO; for macroquad wasm builds this can
# trigger duplicate symbols / bitcode load failures. We explicitly disable LTO here.
RUSTFLAGS="-C target-feature=+simd128 -C lto=no" cargo +nightly build -p quantum_dashboard --target "$TARGET" --release --bin quantum_dashboard_app

WASM_IN="target/$TARGET/release/quantum_dashboard_app.wasm"
WASM_OUT="$DIST/quantum_dashboard_app.wasm"
cp "$WASM_IN" "$WASM_OUT"

# Copy Macroquad's JS loader bundle from the Cargo registry.
MQ_BUNDLE=$(python3 - <<'PY'
import glob
paths = sorted(glob.glob('/home/huuuda/.cargo/registry/src/**/macroquad-0.4.*/js/mq_js_bundle.js', recursive=True))
print(paths[-1] if paths else '')
PY
)

if [ -z "$MQ_BUNDLE" ] || [ ! -f "$MQ_BUNDLE" ]; then
  echo "Could not find mq_js_bundle.js in Cargo registry; attempting to download fallback..." >&2
  FALLBACK_URL="https://raw.githubusercontent.com/not-fl3/macroquad/master/js/mq_js_bundle.js"
  if command -v curl >/dev/null 2>&1; then
    echo "Downloading mq_js_bundle.js from ${FALLBACK_URL}..."
    curl -fsSL "$FALLBACK_URL" -o "$DIST/mq_js_bundle.js" || {
      echo "Failed to download mq_js_bundle.js from fallback URL" >&2
      exit 1
    }
    echo "Downloaded mq_js_bundle.js to $DIST/mq_js_bundle.js"
    MQ_BUNDLE="$DIST/mq_js_bundle.js"
  else
    echo "curl not available to download fallback mq_js_bundle.js" >&2
    echo "Expected under ~/.cargo/registry/src/**/macroquad-0.4.*/js/mq_js_bundle.js" >&2
    exit 1
  fi
fi

# Avoid copying the same file onto itself when the fallback downloaded directly to $DIST
if [ "$MQ_BUNDLE" != "$DIST/mq_js_bundle.js" ]; then
  cp "$MQ_BUNDLE" "$DIST/mq_js_bundle.js"
else
  echo "mq_js_bundle.js already in $DIST; skipping copy"
fi

cp index.html "$DIST/index.html"

if [ -f "telemetry.bin" ]; then
  cp "telemetry.bin" "$DIST/telemetry.bin"
else
  echo "Warning: telemetry.bin not found in repo root (run: cargo run -p generator --release)" >&2
fi

echo "Built dist/:"
ls -la "$DIST"

echo ""
echo "Next: ./serve.sh 8000"
echo "Then open: http://127.0.0.1:8000/"
