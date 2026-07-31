#!/bin/bash
# Cairn -- macOS .pkg build script
# Downloads python-build-standalone, installs a pinned cairn[server],
# and produces a .pkg installer.
#
# Usage: ./build-macos-pkg.sh [VERSION]
#   VERSION: package and cairn version string (default: 1.5.4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/macos"
PAYLOAD_DIR="$BUILD_DIR/payload"
PKG_ID="com.cairn.memory"
PKG_VERSION="${1:-1.5.4}"
CAIRN_VERSION="$PKG_VERSION"
PYTHON_VERSION="3.12"
PYTHON_RELEASE="20250212"

# --- Architecture detection ---
ARCH="$(uname -m)"
case "$ARCH" in
    arm64)  PBS_ARCH="aarch64" ;;
    x86_64) PBS_ARCH="x86_64" ;;
    *)      echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/cpython-${PYTHON_VERSION}.9+${PYTHON_RELEASE}-${PBS_ARCH}-apple-darwin-install_only_stripped.tar.gz"

echo "=== Cairn macOS Installer Build ==="
echo "Architecture: $ARCH (download: $PBS_ARCH)"
echo "Python: $PYTHON_VERSION ($PYTHON_RELEASE)"
echo ""

# --- Clean previous build ---
rm -rf "$BUILD_DIR"
mkdir -p "$PAYLOAD_DIR" "$BUILD_DIR/scripts" "$BUILD_DIR/resources" "$BUILD_DIR/dist"

# --- Step 1: Download python-build-standalone ---
echo "Step 1: Downloading python-build-standalone..."
TARBALL="$BUILD_DIR/python.tar.gz"
curl -fSL --progress-bar -o "$TARBALL" "$PBS_URL"
echo "  Downloaded $(du -h "$TARBALL" | cut -f1)"

# --- Step 2: Extract Python ---
echo "Step 2: Extracting Python..."
tar -xzf "$TARBALL" -C "$PAYLOAD_DIR"
rm "$TARBALL"
echo "  Extracted to $PAYLOAD_DIR/python/"

# --- Step 3: Install cairn[server] ---
echo "Step 3: Installing cairn-memory[server]==$CAIRN_VERSION..."
# The tag triggers this build and the PyPI publish in parallel, and the
# publish now sits behind a ~15-minute test gate (release.yml), so the
# deadline must comfortably outlast the gate (v2.2.0 hit a 10-minute
# deadline while the gate was still running). Bounded, loud on failure.
for attempt in $(seq 1 60); do
  if "$PAYLOAD_DIR/python/bin/python3" -m pip index versions cairn-memory 2>/dev/null | grep -q "$CAIRN_VERSION"; then
    break
  fi
  if [ "$attempt" -eq 60 ]; then
    echo "ERROR: cairn-memory==$CAIRN_VERSION not on PyPI after 30 minutes" >&2
    exit 1
  fi
  echo "  cairn-memory==$CAIRN_VERSION not on PyPI yet (attempt $attempt/60), waiting 30s..."
  sleep 30
done
"$PAYLOAD_DIR/python/bin/python3" -m pip install --quiet --no-cache-dir "cairn-memory[server]==$CAIRN_VERSION"
echo "  Installed cairn==$CAIRN_VERSION"

# --- Step 4: Copy support files ---
echo "Step 4: Copying support files..."
cp "$SCRIPT_DIR/configure_claude.py" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/uninstall-cairn.sh" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/setup-instructions.sh" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/scripts/postinstall" "$BUILD_DIR/scripts/"
cp "$SCRIPT_DIR/macos/resources/"* "$BUILD_DIR/resources/"
cp "$SCRIPT_DIR/macos/Distribution.xml" "$BUILD_DIR/"
python3 - "$BUILD_DIR/Distribution.xml" "$PKG_VERSION" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
version = sys.argv[2]
text = path.read_text()
text = re.sub(
    r'(<pkg-ref id="com\.cairn\.memory"\s+version=")[^"]+(")',
    rf'\g<1>{version}\2',
    text,
    count=1,
)
path.write_text(text)
PY

# --- Step 5: Build component package ---
echo "Step 5: Building component package..."
pkgbuild \
    --identifier "$PKG_ID" \
    --version "$PKG_VERSION" \
    --root "$PAYLOAD_DIR" \
    --install-location "Library/Cairn" \
    --scripts "$BUILD_DIR/scripts" \
    "$BUILD_DIR/cairn.pkg"
echo "  Built component package"

# --- Step 6: Build product archive ---
echo "Step 6: Building product archive..."
productbuild \
    --distribution "$BUILD_DIR/Distribution.xml" \
    --resources "$BUILD_DIR/resources" \
    --package-path "$BUILD_DIR" \
    "$BUILD_DIR/dist/Cairn-Memory.pkg"
echo "  Built Cairn-Memory.pkg"

# --- Done ---
PKG_SIZE="$(du -h "$BUILD_DIR/dist/Cairn-Memory.pkg" | cut -f1)"
echo ""
echo "=== Build complete ==="
echo "Output: $BUILD_DIR/dist/Cairn-Memory.pkg ($PKG_SIZE)"
echo ""
echo "To test: open $BUILD_DIR/dist/Cairn-Memory.pkg"
