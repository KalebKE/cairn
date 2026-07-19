#!/bin/bash
# Cairn -- macOS uninstall script
# Removes Cairn installation while preserving user data (~/.cairn).

set -euo pipefail

INSTALL_DIR="$HOME/Library/Cairn"
PYTHON="$INSTALL_DIR/python/bin/python3"
CONFIGURE="$INSTALL_DIR/configure_claude.py"
PKG_ID="com.cairn.memory"

echo "Cairn Uninstaller"
echo "================="
echo ""

# Step 1: Remove Cairn from Claude Desktop config
echo "Removing Cairn from Claude Desktop configuration..."
if [ -f "$CONFIGURE" ] && [ -f "$PYTHON" ]; then
    "$PYTHON" "$CONFIGURE" --uninstall 2>/dev/null || echo "  (config already clean)"
else
    echo "  Skipped (installer files not found)"
fi

# Step 2: Remove install directory
echo "Removing Cairn installation directory..."
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "  Removed $INSTALL_DIR"
else
    echo "  Already removed"
fi

# Step 3: Forget package receipt
echo "Removing package receipt..."
pkgutil --pkg-info "$PKG_ID" &>/dev/null && pkgutil --forget "$PKG_ID" 2>/dev/null || echo "  No receipt found"

# Step 4: Preserve user data
echo ""
echo "Your Cairn memory data has been preserved at: ~/.cairn"
echo "To remove it permanently: rm -rf ~/.cairn"
echo ""
echo "Cairn has been uninstalled. Restart Claude Desktop."
