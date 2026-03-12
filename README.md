# RetroStream

A cloud retro gaming platform built on libretro cores with WebRTC streaming,
WebSocket input handling, voice chat, and real-time memory watching.

**All Python dependencies are installed via `apt` on Debian 13 (trixie).
No pip, no venv, no Docker required** — though Docker is also supported.

---

## Quick Start (Bare Metal — Debian 13 trixie)

```bash
# 1. Clone / copy the project
cd /opt
git clone <repo> retrostream
cd retrostream

# 2. Install everything (system packages + systemd service)
sudo bash install.sh

# 3. Add cores and ROMs
cp /path/to/snes9x_libretro.so  ./cores/
cp /path/to/game.sfc             ./roms/

# 4. Verify the environment
python3 check_env.py

# 5. Start
python3 -m server
# OR as a background service:
sudo systemctl start retrostream

# 6. Open in browser
# http://localhost:8000
```

---

## What install.sh does

- Installs all apt packages (no pip at all)
- Adds your user to the `video` and `render` groups for VAAPI access
- Installs and registers a `retrostream.service` systemd unit
- Creates the `cores/`, `roms/`, `saves/`, `system/` directories

---

## Architecture

```
aiohttp server  (python3-aiohttp)
├── GET/POST/DELETE /api/sessions     REST session management
├── POST /rtc/offer/{sid}             WebRTC SDP signaling
├── WS   /ws/input/{sid}              Gamepad/keyboard input
├── WS   /ws/memory/{sid}             Memory watch push
└── WS   /ws/chat/{sid}               Text chat

Per-session (one per ROM instance):
├── LibretroCore     ctypes wrapper around .so core
├── SessionEncoder   FFmpeg VAAPI/QSV/software H.264 subprocess
├── LibretroVideoTrack  aiortc track fed from encoder
├── MemoryWatcher    async poller → WebSocket push
└── InputMerger      4-player input bitmask merging
```

---

## Debian 13 Package Map

| Functionality     | apt package          |
|-------------------|----------------------|
| Web server + WS   | `python3-aiohttp`    |
| WebRTC            | `python3-aiortc`     |
| WebSockets        | `python3-websockets` |
| Numeric arrays    | `python3-numpy`      |
| YAML parsing      | `python3-yaml`       |
| Event emitter     | `python3-pyee`       |
| Video encode      | `ffmpeg`             |
| VAAPI driver      | `intel-media-va-driver` |
| GPU monitoring    | `intel-gpu-tools`    |

---

## Memory Map Format

Add a YAML file to `configs/memory_maps/<rom_name>.yml`:

```yaml
lives:
  addr: 0x0DBE
  type: u8

score:
  addr: 0x0F34
  type: u24
  endian: little

powerup:
  addr: 0x0019
  type: u8
  bitmask: 0x0F
```

Supported types: `u8`, `s8`, `u16`, `s16`, `u24`, `u32`, `s32`

---

## Checking Intel VAAPI / QuickSync

```bash
vainfo                       # list supported encode/decode profiles
ffmpeg -hwaccels             # confirm vaapi is listed
ls /dev/dri/                 # renderD128 should exist
intel_gpu_top                # live GPU utilisation
```

If VAAPI is unavailable, the encoder automatically falls back to `libx264`
(software). Set `ENCODE_BACKEND=software` to force this.

---

## Service Management

```bash
sudo systemctl start   retrostream   # start
sudo systemctl stop    retrostream   # stop
sudo systemctl enable  retrostream   # start on boot
sudo systemctl status  retrostream   # check status
journalctl -u retrostream -f         # live logs
```

---

## Docker (optional)

```bash
docker compose up
```

Docker Compose mounts `/dev/dri/renderD128` automatically. See
`docker-compose.yml` for GPU group configuration.
