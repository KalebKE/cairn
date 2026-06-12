#!/bin/bash
# OMEGA Memory -- macOS .pkg build script
# Downloads python-build-standalone, installs a pinned omega-memory[server],
# and produces a .pkg installer.
#
# Usage: ./build-macos-pkg.sh [VERSION]
#   VERSION: package and omega-memory version string (default: 1.5.4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/macos"
PAYLOAD_DIR="$BUILD_DIR/payload"
PKG_ID="com.omega.memory"
PKG_VERSION="${1:-1.5.4}"
OMEGA_VERSION="$PKG_VERSION"
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

echo "=== OMEGA macOS Installer Build ==="
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

# --- Step 3: Install omega-memory[server] ---
echo "Step 3: Installing omega-memory[server]==$OMEGA_VERSION..."
"$PAYLOAD_DIR/python/bin/python3" -m pip install --quiet --no-cache-dir "omega-memory[server]==$OMEGA_VERSION"
echo "  Installed omega-memory==$OMEGA_VERSION"

# --- Step 4: Copy support files ---
echo "Step 4: Copying support files..."
cp "$SCRIPT_DIR/configure_claude.py" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/uninstall-omega.sh" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/setup-instructions.sh" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/scripts/postinstall" "$BUILD_DIR/scripts/"
cp "$SCRIPT_DIR/macos/resources/"* "$BUILD_DIR/resources/"
cp "$SCRIPT_DIR/macos/Distribution.xml" "$BUILD_DIR/"

# --- Step 5: Build component package ---
echo "Step 5: Building component package..."
pkgbuild \
    --identifier "$PKG_ID" \
    --version "$PKG_VERSION" \
    --root "$PAYLOAD_DIR" \
    --install-location "Library/OMEGA" \
    --scripts "$BUILD_DIR/scripts" \
    "$BUILD_DIR/omega-memory.pkg"
echo "  Built component package"

# --- Step 6: Build product archive ---
echo "Step 6: Building product archive..."
productbuild \
    --distribution "$BUILD_DIR/Distribution.xml" \
    --resources "$BUILD_DIR/resources" \
    --package-path "$BUILD_DIR" \
    "$BUILD_DIR/dist/OMEGA-Memory.pkg"
echo "  Built OMEGA-Memory.pkg"

# --- Done ---
PKG_SIZE="$(du -h "$BUILD_DIR/dist/OMEGA-Memory.pkg" | cut -f1)"
echo ""
echo "=== Build complete ==="
echo "Output: $BUILD_DIR/dist/OMEGA-Memory.pkg ($PKG_SIZE)"
echo ""
echo "To test: open $BUILD_DIR/dist/OMEGA-Memory.pkg"
