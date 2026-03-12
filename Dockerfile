# Use Debian 13 (trixie) as base — same as target bare-metal system
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive

# All deps from apt — no pip, no venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-aiohttp \
    python3-aiortc \
    python3-websockets \
    python3-numpy \
    python3-yaml \
    python3-pyee \
    ffmpeg \
    libva-drm2 \
    libva2 \
    vainfo \
    intel-gpu-tools \
    intel-media-va-driver \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/ ./server/
COPY frontend/ ./frontend/
COPY configs/ ./configs/

RUN mkdir -p cores roms saves system

ENV VAAPI_DEVICE=/dev/dri/renderD128
ENV ENCODE_BACKEND=auto
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "server"]
