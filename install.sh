#!/usr/bin/env bash
# =============================================================================
# RetroStream — Bare-metal installer for Debian 13 (trixie)
#
# Installs all dependencies from apt. No pip, no venv, no Docker required.
# Run as root or with sudo.
#
# Usage:
#   sudo bash install.sh
# =============================================================================

set -euo pipefail

RETROSTREAM_USER="${SUDO_USER:-$(whoami)}"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[[ $EUID -eq 0 ]] || error "Run with sudo: sudo bash install.sh"

info "RetroStream installer — Debian 13 (trixie)"
info "Install directory: $INSTALL_DIR"
info "Running as user:   $RETROSTREAM_USER"

# ── 1. APT packages ───────────────────────────────────────────────────────────
info "Updating apt..."
apt-get update -qq

info "Installing system packages..."
apt-get install -y --no-install-recommends \
    \
    `# Python runtime` \
    python3 \
    python3-fastapi \
    python3-uvicorn \
    python3-aiortc \
    python3-websockets \
    python3-numpy \
    python3-yaml \
    python3-pyee \
    python3-aiofiles \
    python3-multipart \
    python3-pydantic \
    \
    `# FFmpeg with VAAPI/QSV support` \
    ffmpeg \
    \
    `# Intel GPU / VAAPI` \
    libva-drm2 \
    libva2 \
    vainfo \
    intel-gpu-tools \
    i965-va-driver \
    intel-media-va-driver \
    \
    `# libretro core runtime deps` \
    libgl1 \
    libglib2.0-0 \
    \
    `# Utilities` \
    curl \
    ca-certificates

info "APT packages installed."

# ── 2. VAAPI group membership ─────────────────────────────────────────────────
info "Checking GPU/VAAPI availability..."
if [[ -e /dev/dri/renderD128 ]]; then
    info "Found /dev/dri/renderD128"
    usermod -aG video,render "$RETROSTREAM_USER" 2>/dev/null || true
    info "Added $RETROSTREAM_USER to video/render groups (re-login to take effect)"
else
    warn "/dev/dri/renderD128 not found — encoder will fall back to software (libx264)"
fi

# ── 3. Runtime directories ────────────────────────────────────────────────────
info "Creating runtime directories..."
for d in cores roms saves system configs/memory_maps; do
    mkdir -p "$INSTALL_DIR/$d"
    chown "$RETROSTREAM_USER:$RETROSTREAM_USER" "$INSTALL_DIR/$d"
done

# ── 4. Systemd service ────────────────────────────────────────────────────────
info "Installing systemd service..."
cat > /etc/systemd/system/retrostream.service << EOF
[Unit]
Description=RetroStream — Retro Gaming Cloud Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=${RETROSTREAM_USER}
WorkingDirectory=${INSTALL_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=VAAPI_DEVICE=/dev/dri/renderD128
Environment=ENCODE_BACKEND=auto
ExecStart=/usr/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SupplementaryGroups=video render

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
info "Systemd service installed: retrostream.service"

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  RetroStream installation complete!          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Next steps:"
echo "  1. Place libretro cores (.so files) in:  $INSTALL_DIR/cores/"
echo "  2. Place ROM files in:                   $INSTALL_DIR/roms/"
echo "  3. Verify environment:   python3 check_env.py"
echo "  4. Start manually:       uvicorn server.main:app --host 0.0.0.0 --port 8000"
echo "     OR as a service:      sudo systemctl start retrostream"
echo "     Enable on boot:       sudo systemctl enable retrostream"
echo ""
echo "  Logs:        journalctl -u retrostream -f"
echo "  API docs:    http://localhost:8000/docs"
echo "  Frontend:    http://localhost:8000"
echo ""
if [[ -e /dev/dri/renderD128 ]]; then
    echo -e "  GPU: ${GREEN}Intel VAAPI detected — hardware encode enabled${NC}"
else
    echo -e "  GPU: ${YELLOW}No VAAPI device found — software encode (libx264) will be used${NC}"
fi
echo ""
warn "If you were added to video/render groups, log out and back in first."
