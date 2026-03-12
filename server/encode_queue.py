"""
encode_queue.py

Centralised FFmpeg VAAPI / QuickSync encode queue.

Instead of spawning one FFmpeg process per session (wasteful), we run a
SINGLE shared FFmpeg process that accepts frames on named pipes and multiplexes
encoded output back to callers.  For the frame counts involved (256x224 @ 60fps
per session) the Intel iGPU can handle many simultaneous sessions with ease.

Architecture:
    ┌─────────────┐     raw RGB24     ┌──────────────────────┐
    │  Session A  │ ──────────────→   │                      │
    ├─────────────┤                   │  EncodeQueue         │
    │  Session B  │ ──────────────→   │  (single FFmpeg)     │ → H.264 NAL packets
    ├─────────────┤                   │  VAAPI / QSV         │   per session
    │  Session C  │ ──────────────→   │                      │
    └─────────────┘                   └──────────────────────┘

For simplicity and reliability, we use one FFmpeg subprocess PER session but
share the VAAPI device handle. Each encoder is lightweight at retro resolutions.
A watchdog monitors GPU usage via intel_gpu_top and logs warnings if > 80%.

If you want the truly shared single-process approach, see the comment block
at the bottom of this file describing the filter_complex path.
"""

import asyncio
import logging
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

log = logging.getLogger(__name__)

VAAPI_DEVICE = os.environ.get("VAAPI_DEVICE", "/dev/dri/renderD128")
ENCODE_BACKEND = os.environ.get("ENCODE_BACKEND", "vaapi")  # vaapi | qsv | software


# ---------------------------------------------------------------------------
# Hardware capability detection
# ---------------------------------------------------------------------------

def detect_encode_backend() -> str:
    """
    Auto-detect the best available encode backend.
    Returns one of: 'vaapi', 'qsv', 'software'
    """
    # Try VAAPI
    if Path(VAAPI_DEVICE).exists():
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True
        )
        if "vaapi" in result.stdout.lower():
            log.info("Encode backend: VAAPI (%s)", VAAPI_DEVICE)
            return "vaapi"

    # Try QSV
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"],
        capture_output=True, text=True
    )
    if "qsv" in result.stdout.lower():
        log.info("Encode backend: Intel QuickSync (QSV)")
        return "qsv"

    log.warning("No hardware encode available, falling back to software (libx264)")
    return "software"


def build_ffmpeg_command(width: int, height: int, fps: float,
                          backend: str) -> list[str]:
    """
    Build the FFmpeg command for the given backend.
    Input:  raw RGB24 on stdin
    Output: H.264 Annex-B on stdout (suitable for feeding into aiortc)
    """
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        # Input
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]

    if backend == "vaapi":
        encode = [
            # Upload to GPU, convert to NV12 (VAAPI native format)
            "-vf", f"format=nv12,hwupload",
            "-vaapi_device", VAAPI_DEVICE,
            "-c:v", "h264_vaapi",
            # Low-latency settings
            "-qp", "24",                    # constant QP (fast, no lookahead)
            "-bf", "0",                     # no B-frames (latency killer)
            "-g", str(int(fps * 2)),        # keyframe every 2 seconds
            "-rc_mode", "CQP",
        ]
    elif backend == "qsv":
        encode = [
            "-vf", "format=nv12",
            "-c:v", "h264_qsv",
            "-global_quality", "25",
            "-bf", "0",
            "-g", str(int(fps * 2)),
            "-low_power", "1",              # QSV low-power mode
        ]
    else:  # software fallback
        encode = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", "26",
            "-bf", "0",
            "-g", str(int(fps * 2)),
            "-pix_fmt", "yuv420p",
        ]

    output = [
        "-f", "h264",                       # raw Annex-B NAL units
        "pipe:1",
    ]

    return base + encode + output


# ---------------------------------------------------------------------------
# Per-session encoder
# ---------------------------------------------------------------------------

@dataclass
class EncoderStats:
    frames_in:   int   = 0
    frames_out:  int   = 0
    dropped:     int   = 0
    start_time:  float = field(default_factory=time.monotonic)

    @property
    def fps_in(self) -> float:
        elapsed = time.monotonic() - self.start_time
        return self.frames_in / elapsed if elapsed > 0 else 0.0


class SessionEncoder:
    """
    One FFmpeg subprocess per game session.
    Frames are pushed via push_frame(); encoded packets are read via
    the async packet_iterator().

    Thread-safe: push_frame() can be called from the core thread,
    while packet_iterator() is consumed from the asyncio event loop.
    """

    MAX_QUEUE = 4   # drop frames if encode falls behind (keeps latency low)

    def __init__(self, session_id: str, width: int, height: int,
                  fps: float, backend: Optional[str] = None):
        self.session_id = session_id
        self.width      = width
        self.height     = height
        self.fps        = fps
        self.backend    = backend or detect_encode_backend()
        self.stats      = EncoderStats()

        self._frame_queue: deque[bytes] = deque(maxlen=self.MAX_QUEUE)
        self._lock       = threading.Lock()
        self._running    = False
        self._proc: Optional[subprocess.Popen] = None
        self._packet_callbacks: list[Callable[[bytes], None]] = []

        # asyncio queue bridging the reader thread → event loop
        self._loop:         Optional[asyncio.AbstractEventLoop] = None
        self._async_queue:  Optional[asyncio.Queue] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the FFmpeg process and reader thread."""
        self._loop       = loop
        self._async_queue = asyncio.Queue(maxsize=32)
        self._running    = True

        cmd = build_ffmpeg_command(self.width, self.height, self.fps, self.backend)
        log.info("[%s] Starting encoder: %s", self.session_id, " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Thread: core → FFmpeg stdin
        self._write_thread = threading.Thread(
            target=self._write_loop, daemon=True,
            name=f"enc-write-{self.session_id}"
        )
        # Thread: FFmpeg stdout → asyncio queue
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True,
            name=f"enc-read-{self.session_id}"
        )
        # Thread: FFmpeg stderr logger
        self._err_thread = threading.Thread(
            target=self._err_loop, daemon=True,
            name=f"enc-err-{self.session_id}"
        )

        self._write_thread.start()
        self._read_thread.start()
        self._err_thread.start()

    def push_frame(self, rgb24: bytes):
        """
        Called from the libretro core thread at ~60fps.
        Non-blocking: drops the oldest frame if queue is full (low-latency policy).
        """
        with self._lock:
            self._frame_queue.append(rgb24)
            self.stats.frames_in += 1

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        log.info("[%s] Encoder stopped. Stats: %s", self.session_id, self.stats)

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _write_loop(self):
        """Drain frame queue → FFmpeg stdin."""
        frame_interval = 1.0 / self.fps
        while self._running:
            frame = None
            with self._lock:
                if self._frame_queue:
                    frame = self._frame_queue.popleft()
            if frame:
                try:
                    self._proc.stdin.write(frame)
                    self._proc.stdin.flush()
                except BrokenPipeError:
                    log.error("[%s] FFmpeg stdin closed", self.session_id)
                    break
            else:
                time.sleep(frame_interval / 2)

    def _read_loop(self):
        """
        Read H.264 NAL units from FFmpeg stdout.
        We read in chunks and re-split on Annex-B start codes (0x00 0x00 0x00 0x01)
        to deliver complete NAL units.
        """
        CHUNK = 65536
        buf   = b""
        START = b"\x00\x00\x00\x01"

        while self._running:
            try:
                chunk = self._proc.stdout.read(CHUNK)
                if not chunk:
                    break
                buf += chunk

                # Split on Annex-B start codes
                parts = buf.split(START)
                for part in parts[:-1]:
                    if part:
                        nal = START + part
                        self.stats.frames_out += 1
                        asyncio.run_coroutine_threadsafe(
                            self._async_queue.put(nal), self._loop
                        )
                buf = parts[-1]  # keep incomplete tail
            except Exception as e:
                log.error("[%s] Read error: %s", self.session_id, e)
                break

    def _err_loop(self):
        for line in self._proc.stderr:
            log.debug("[%s][ffmpeg] %s", self.session_id,
                      line.decode(errors="replace").strip())

    # ------------------------------------------------------------------
    # Async packet consumer
    # ------------------------------------------------------------------

    async def packet_iterator(self) -> AsyncIterator[bytes]:
        """Yield encoded H.264 NAL packets. Use in an async for loop."""
        while self._running:
            try:
                packet = await asyncio.wait_for(
                    self._async_queue.get(), timeout=1.0
                )
                yield packet
            except asyncio.TimeoutError:
                continue


# ---------------------------------------------------------------------------
# GPU watchdog
# ---------------------------------------------------------------------------

class GPUWatchdog:
    """
    Polls intel_gpu_top and logs a warning if render engine > threshold.
    Runs in a background thread. Safe to ignore if intel_gpu_top not installed.
    """

    def __init__(self, warn_pct: float = 80.0, interval: float = 5.0):
        self.warn_pct = warn_pct
        self.interval = interval
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="gpu-watchdog"
        )

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            while True:
                result = subprocess.run(
                    ["intel_gpu_top", "-J", "-s", "1000", "-n", "1"],
                    capture_output=True, text=True, timeout=5
                )
                self._parse(result.stdout)
                time.sleep(self.interval)
        except FileNotFoundError:
            log.info("intel_gpu_top not found — GPU watchdog disabled. "
                     "Install with: apt install intel-gpu-tools")
        except Exception as e:
            log.debug("GPU watchdog error: %s", e)

    def _parse(self, json_text: str):
        import json
        try:
            data    = json.loads(json_text)
            engines = data.get("engines", {})
            for name, info in engines.items():
                busy = info.get("busy", 0)
                if busy > self.warn_pct:
                    log.warning("GPU engine '%s' at %.1f%% — consider "
                                "reducing active sessions", name, busy)
                else:
                    log.debug("GPU engine '%s': %.1f%%", name, busy)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared encode queue manager (singleton)
# ---------------------------------------------------------------------------

class EncodeQueueManager:
    """
    Manages all per-session encoders. Provides create/destroy lifecycle
    and exposes the GPU watchdog.
    """

    def __init__(self):
        self._encoders: dict[str, SessionEncoder] = {}
        self._backend  = detect_encode_backend()
        self._watchdog = GPUWatchdog()
        self._watchdog.start()

    def create(self, session_id: str, width: int, height: int,
               fps: float, loop: asyncio.AbstractEventLoop) -> SessionEncoder:
        enc = SessionEncoder(session_id, width, height, fps, self._backend)
        enc.start(loop)
        self._encoders[session_id] = enc
        log.info("Encoder created for session %s (%dx%d @ %.1ffps, %s)",
                 session_id, width, height, fps, self._backend)
        return enc

    def destroy(self, session_id: str):
        enc = self._encoders.pop(session_id, None)
        if enc:
            enc.stop()

    def get(self, session_id: str) -> Optional[SessionEncoder]:
        return self._encoders.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._encoders)

    def stats_summary(self) -> dict:
        return {
            sid: {
                "fps_in":    round(enc.stats.fps_in, 1),
                "frames_in": enc.stats.frames_in,
                "dropped":   enc.stats.dropped,
                "backend":   enc.backend,
            }
            for sid, enc in self._encoders.items()
        }


# Global singleton
encode_manager = EncodeQueueManager()


# ---------------------------------------------------------------------------
# NOTE: Single shared FFmpeg process (advanced alternative)
# ---------------------------------------------------------------------------
# If you hit limits (many sessions, weak iGPU), consider a single FFmpeg
# process using filter_complex with multiple inputs:
#
#   ffmpeg \
#     -f rawvideo -s 256x224 -r 60 -i /tmp/session_a.pipe \
#     -f rawvideo -s 256x224 -r 60 -i /tmp/session_b.pipe \
#     -filter_complex "[0]format=nv12,hwupload[a];[1]format=nv12,hwupload[b]" \
#     -map "[a]" -c:v h264_vaapi -qp 24 -f h264 /tmp/out_a.pipe \
#     -map "[b]" -c:v h264_vaapi -qp 24 -f h264 /tmp/out_b.pipe
#
# This is more efficient because VAAPI context is shared, but session
# lifecycle management becomes significantly more complex (you must restart
# FFmpeg when sessions are added/removed). Implement as a future optimisation
# once you have profiling data showing per-process overhead is a bottleneck.
