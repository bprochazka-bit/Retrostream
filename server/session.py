"""
session.py

A GameSession owns one libretro core process, the WebRTC peer connections
for all players/spectators, and a MemoryWatcher.

Lifecycle:
    session = await GameSession.create(config)
    await session.start()
    ...
    await session.stop()
"""

import asyncio
import json
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    AudioStreamTrack,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaStreamError
import av
import fractions
import numpy as np

from .libretro_core import LibretroCore, RETRO_MEMORY_SYSTEM_RAM
from .memory_watcher import MemoryWatcher, load_memory_map_for_rom

log = logging.getLogger(__name__)


class PlayerRole(str, Enum):
    PLAYER    = "player"
    SPECTATOR = "spectator"


@dataclass
class SessionConfig:
    core_path:   str
    rom_path:    str
    max_players: int = 4
    memory_maps_dir: str = "configs/memory_maps"
    system_dir:  str = "./system"
    save_dir:    str = "./saves"


@dataclass
class ConnectedClient:
    client_id: str
    role:      PlayerRole
    player_num: Optional[int]          # 0-3 for players, None for spectators
    pc:        RTCPeerConnection
    input_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(32))


# ---------------------------------------------------------------------------
# aiortc video track — delivers raw av.VideoFrame from libretro core
# ---------------------------------------------------------------------------

class LibretroVideoTrack(VideoStreamTrack):
    """
    Receives raw RGB24 frames from the libretro core thread and delivers
    them as av.VideoFrame objects to aiortc, which encodes them for WebRTC.
    """

    kind = "video"

    def __init__(self, width: int, height: int, fps: float,
                 loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._width     = width
        self._height    = height
        self._fps       = fps
        self._loop      = loop
        self._pts       = 0
        self._time_base = fractions.Fraction(1, 90000)
        # Latest frame buffer (written from core thread, read from asyncio)
        self._latest_frame: Optional[bytes] = None
        self._frame_event = asyncio.Event()
        self._stopped = False
        self._started = False  # send black frame immediately on first recv

    def push_frame(self, rgb24: bytes):
        """Called from the core thread. Stores frame and signals waiters."""
        self._latest_frame = rgb24
        try:
            self._loop.call_soon_threadsafe(self._frame_event.set)
        except RuntimeError:
            pass  # loop closed

    async def recv(self):
        """Called by aiortc to get the next video frame."""
        if not self._started:
            # Send a black frame immediately so WebRTC negotiation completes fast
            self._started = True
            frame = av.VideoFrame(self._width, self._height, 'rgb24')
            frame.pts = self._pts
            frame.time_base = self._time_base
            self._pts += int(90000 / self._fps)
            return frame

        # Wait for a real frame from the core
        while not self._stopped:
            try:
                await asyncio.wait_for(self._frame_event.wait(), timeout=0.5)
                break
            except asyncio.TimeoutError:
                if self._latest_frame is not None:
                    break
                # No frame yet — return last or black
                frame = av.VideoFrame(self._width, self._height, 'rgb24')
                frame.pts = self._pts
                frame.time_base = self._time_base
                self._pts += int(90000 / self._fps)
                return frame

        self._frame_event.clear()
        rgb24 = self._latest_frame

        if rgb24 is None:
            frame = av.VideoFrame(self._width, self._height, 'rgb24')
        else:
            array = np.frombuffer(rgb24, dtype=np.uint8).reshape(
                (self._height, self._width, 3)
            )
            frame = av.VideoFrame.from_ndarray(array, format='rgb24')

        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += int(90000 / self._fps)
        return frame

    async def stop(self):
        self._stopped = True
        self._frame_event.set()  # unblock any waiting recv()
        await super().stop()


# ---------------------------------------------------------------------------
# aiortc audio track — delivers PCM from libretro core as av.AudioFrame
# ---------------------------------------------------------------------------

class LibretroAudioTrack(AudioStreamTrack):
    """
    Receives interleaved int16 stereo PCM from the libretro core thread
    and delivers it as av.AudioFrame objects to aiortc.
    """

    kind = "audio"

    def __init__(self, sample_rate: int, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._sample_rate = sample_rate
        self._loop        = loop
        self._pts         = 0
        self._time_base   = fractions.Fraction(1, sample_rate)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._stopped = False
        self._recv_count = 0
        self._push_count = 0

    def push_audio(self, pcm: bytes):
        """Called from core thread. pcm is interleaved int16 stereo."""
        self._push_count += 1
        if self._push_count <= 5:
            log.info("AudioTrack: push_audio #%d, %d bytes", self._push_count, len(pcm))
        try:
            self._loop.call_soon_threadsafe(self._queue_put, pcm)
        except RuntimeError:
            pass

    def _queue_put(self, pcm: bytes):
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            # Drop oldest to keep latency low
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(pcm)
            except asyncio.QueueFull:
                pass

    async def recv(self):
        """Called by aiortc to get the next audio frame."""
        self._recv_count += 1
        if self._recv_count <= 5:
            log.info("AudioTrack: recv() called #%d, queue size=%d",
                     self._recv_count, self._queue.qsize())
        try:
            pcm = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Generate silence (960 samples = 20ms at 48kHz, common WebRTC frame)
            samples = 960
            pcm = b'\x00' * (samples * 2 * 2)  # stereo int16

        # Convert to numpy array: interleaved int16 stereo
        samples_array = np.frombuffer(pcm, dtype=np.int16)
        num_samples = len(samples_array) // 2  # stereo → per-channel count

        if num_samples == 0:
            num_samples = 960
            samples_array = np.zeros(num_samples * 2, dtype=np.int16)

        # Pack as (1, total_samples) for av s16 packed format (required by Opus encoder)
        packed = samples_array.reshape(1, -1)  # shape: (1, num_samples*2)

        frame = av.AudioFrame.from_ndarray(
            packed, format='s16', layout='stereo'
        )
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += num_samples
        return frame

    async def stop(self):
        self._stopped = True
        await super().stop()


# ---------------------------------------------------------------------------
# Game session
# ---------------------------------------------------------------------------

class GameSession:

    def __init__(self, session_id: str, config: SessionConfig):
        self.session_id = session_id
        self.config     = config
        self.created_at = time.time()

        self._core:    Optional[LibretroCore]    = None
        self._watcher: Optional[MemoryWatcher]  = None
        self._clients: dict[str, ConnectedClient] = {}
        self._loop:    Optional[asyncio.AbstractEventLoop] = None

        # Core runs in a dedicated thread
        self._core_thread: Optional[threading.Thread] = None
        self._running = False

        # Input state: player_num → button bitmask
        self._inputs: dict[int, int] = {i: 0 for i in range(4)}
        self._input_lock = threading.Lock()

        # Shared tracks for all clients
        self._video_track: Optional[LibretroVideoTrack] = None
        self._audio_track: Optional[LibretroAudioTrack] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, config: SessionConfig) -> "GameSession":
        sid     = str(uuid.uuid4())[:8]
        session = cls(sid, config)
        return session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._running = True

        # 1. Init libretro core
        self._core = LibretroCore(
            self.config.core_path,
            system_dir=self.config.system_dir,
            save_dir=self.config.save_dir,
        )
        self._core.on_frame = self._on_frame
        self._core.on_audio = self._on_audio
        self._core.init()

        if not self._core.load_game(self.config.rom_path):
            raise RuntimeError(f"Failed to load ROM: {self.config.rom_path}")

        # 2. Create shared video and audio tracks
        self._video_track = LibretroVideoTrack(
            self._core.width,
            self._core.height,
            self._core.fps,
            self._loop,
        )
        self._audio_track = LibretroAudioTrack(
            int(self._core.sample_rate),
            self._loop,
        )
        log.info("Session %s: video=%dx%d@%.1ffps audio=%dHz",
                 self.session_id, self._core.width, self._core.height,
                 self._core.fps, int(self._core.sample_rate))

        # 3. Memory watcher
        mmap = load_memory_map_for_rom(
            self.config.rom_path, self.config.memory_maps_dir
        )
        self._watcher = MemoryWatcher(
            self.session_id,
            lambda: self._core.get_memory(RETRO_MEMORY_SYSTEM_RAM),
            mmap,
        )
        await self._watcher.start()

        # 4. Start core thread
        self._core_thread = threading.Thread(
            target=self._core_loop, daemon=True,
            name=f"core-{self.session_id}"
        )
        self._core_thread.start()

        log.info("Session %s started: %s / %s",
                 self.session_id,
                 Path(self.config.core_path).name,
                 Path(self.config.rom_path).name)

    async def stop(self):
        self._running = False

        if self._watcher:
            await self._watcher.stop()

        # Stop tracks
        if self._video_track:
            await self._video_track.stop()
        if self._audio_track:
            await self._audio_track.stop()

        # Close all peer connections
        for client in list(self._clients.values()):
            await client.pc.close()
        self._clients.clear()

        if self._core:
            self._core.unload()

        if self._core_thread:
            self._core_thread.join(timeout=2)

        log.info("Session %s stopped", self.session_id)

    # ------------------------------------------------------------------
    # Core thread (runs at target FPS)
    # ------------------------------------------------------------------

    def _core_loop(self):
        """Runs the libretro core at native FPS in a dedicated thread."""
        target_fps  = self._core.fps
        frame_time  = 1.0 / target_fps
        next_frame  = time.monotonic()
        frame_num   = 0

        log.info("[%s] Core loop starting: fps=%.2f frame_time=%.4fs",
                 self.session_id, target_fps, frame_time)

        while self._running:
            # Apply merged inputs
            with self._input_lock:
                for player, buttons in self._inputs.items():
                    self._core.set_input(player, buttons)

            frame_num += 1
            if frame_num <= 5:
                log.info("[%s] Running frame #%d...", self.session_id, frame_num)

            try:
                self._core.run()
            except Exception as e:
                log.error("[%s] Exception on frame #%d: %s", self.session_id, frame_num, e)
                self._running = False
                break

            if frame_num <= 5:
                log.info("[%s] Frame #%d complete", self.session_id, frame_num)

            # Precise frame pacing
            next_frame += frame_time
            sleep_for = next_frame - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -frame_time * 3:
                # We're more than 3 frames behind — reset pacing
                next_frame = time.monotonic()

    # ------------------------------------------------------------------
    # Core callbacks (called from core thread)
    # ------------------------------------------------------------------

    def _on_frame(self, rgb24: bytes, width: int, height: int):
        if self._video_track:
            self._video_track.push_frame(rgb24)

    def _on_audio(self, pcm: bytes):
        if self._audio_track:
            self._audio_track.push_audio(pcm)

    # ------------------------------------------------------------------
    # WebRTC
    # ------------------------------------------------------------------

    async def handle_offer(self, sdp: str, sdp_type: str,
                           client_id: str, role: str) -> dict:
        """
        Process a WebRTC offer from a client.
        Returns {"sdp": ..., "type": "answer", "client_id": ...}
        """
        player_role = PlayerRole(role)
        player_num  = None

        if player_role == PlayerRole.PLAYER:
            player_num = self._next_player_slot()
            if player_num is None:
                raise ValueError("Session is full")

        pc = RTCPeerConnection()

        # Add the shared video and audio tracks
        pc.addTrack(self._video_track)
        pc.addTrack(self._audio_track)

        # Handle ICE connection state
        @pc.on("connectionstatechange")
        async def on_state():
            log.info("[%s] Client %s: %s", self.session_id, client_id,
                     pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._remove_client(client_id)

        # Handle input DataChannel (low-latency UDP/SCTP from client)
        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label == "input":
                log.info("[%s] Input DataChannel opened for %s",
                         self.session_id, client_id)

                @channel.on("message")
                def on_input_message(message):
                    try:
                        if isinstance(message, bytes) and len(message) >= 2:
                            buttons = int.from_bytes(message[:2], "little")
                        else:
                            msg = json.loads(message)
                            buttons = int(msg.get("buttons", 0))
                        self.apply_input(client_id, buttons)
                    except Exception as e:
                        log.debug("Bad DC input: %s", e)

        # Handle incoming audio (voice chat) from client
        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                log.info("[%s] Audio track from %s", self.session_id, client_id)
                asyncio.create_task(
                    self._relay_audio(client_id, track)
                )

        client = ConnectedClient(
            client_id  = client_id,
            role       = player_role,
            player_num = player_num,
            pc         = pc,
        )
        self._clients[client_id] = client

        # Exchange SDP
        offer  = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Log SDP details for debugging
        answer_sdp = pc.localDescription.sdp
        audio_lines = [l for l in answer_sdp.split('\n') if 'audio' in l.lower() or 'm=audio' in l]
        video_lines = [l for l in answer_sdp.split('\n') if 'm=video' in l]
        log.info("[%s] SDP answer: %d video m-lines, %d audio m-lines",
                 self.session_id, len(video_lines), len(audio_lines))
        for line in audio_lines:
            log.info("[%s] SDP audio: %s", self.session_id, line.strip())

        log.info("[%s] Client %s joined as %s (player %s)",
                 self.session_id, client_id, role, player_num)

        return {
            "sdp":       pc.localDescription.sdp,
            "type":      pc.localDescription.type,
            "client_id": client_id,
            "player_num": player_num,
        }

    # ------------------------------------------------------------------
    # Voice relay
    # ------------------------------------------------------------------

    async def _relay_audio(self, sender_id: str, track: MediaStreamTrack):
        """Relay audio from one client to all other clients."""
        while True:
            try:
                frame = await track.recv()
                for cid, client in self._clients.items():
                    if cid != sender_id:
                        pass
            except MediaStreamError:
                break

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def reset_game(self):
        """Reset the running game (like pressing the reset button on a console)."""
        if self._core:
            self._core.reset()
            log.info("[%s] Game reset", self.session_id)

    def apply_input(self, client_id: str, buttons: int):
        """
        Apply a button bitmask from a client.
        Maps client → their assigned player slot.
        """
        client = self._clients.get(client_id)
        if client and client.player_num is not None:
            with self._input_lock:
                self._inputs[client.player_num] = buttons

    def set_player_num(self, client_id: str, player_num: int) -> bool:
        """
        Change which player slot a client controls (0-3).
        Returns True if successful, False if slot is taken or invalid.
        """
        if player_num < 0 or player_num >= self.config.max_players:
            return False
        client = self._clients.get(client_id)
        if not client:
            return False
        # Check if slot is already taken by another client
        for cid, c in self._clients.items():
            if cid != client_id and c.player_num == player_num:
                return False
        # Clear old slot
        if client.player_num is not None:
            with self._input_lock:
                self._inputs[client.player_num] = 0
        client.player_num = player_num
        log.info("[%s] Client %s now controls player %d",
                 self.session_id, client_id, player_num)
        return True

    # ------------------------------------------------------------------
    # Memory watch
    # ------------------------------------------------------------------

    def subscribe_memory(self) -> asyncio.Queue:
        return self._watcher.subscribe()

    def unsubscribe_memory(self, q: asyncio.Queue):
        self._watcher.unsubscribe(q)

    def memory_snapshot(self) -> dict:
        return self._watcher.snapshot()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_player_slot(self) -> Optional[int]:
        used = {c.player_num for c in self._clients.values()
                if c.player_num is not None}
        for i in range(self.config.max_players):
            if i not in used:
                return i
        return None

    async def _remove_client(self, client_id: str):
        client = self._clients.pop(client_id, None)
        if client:
            if client.player_num is not None:
                with self._input_lock:
                    self._inputs[client.player_num] = 0
            log.info("[%s] Client %s removed", self.session_id, client_id)

    @property
    def player_count(self) -> int:
        return sum(1 for c in self._clients.values()
                   if c.role == PlayerRole.PLAYER)

    @property
    def spectator_count(self) -> int:
        return sum(1 for c in self._clients.values()
                   if c.role == PlayerRole.SPECTATOR)

    def info(self) -> dict:
        return {
            "session_id":  self.session_id,
            "rom":         Path(self.config.rom_path).name,
            "core":        Path(self.config.core_path).name,
            "players":     self.player_count,
            "spectators":  self.spectator_count,
            "max_players": self.config.max_players,
            "created_at":  self.created_at,
        }
