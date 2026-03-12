"""
session.py

A GameSession owns one libretro core process, one FFmpeg encoder,
one MemoryWatcher, and the WebRTC peer connections for all players/spectators.

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
from .encode_queue import encode_manager, SessionEncoder
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
# aiortc video track that consumes NAL packets from the encoder
# ---------------------------------------------------------------------------

class LibretroVideoTrack(VideoStreamTrack):
    """
    Feeds encoded H.264 NAL units into aiortc as video frames.
    aiortc will packetize these into RTP for WebRTC.
    """

    kind = "video"

    def __init__(self, encoder: SessionEncoder):
        super().__init__()
        self._encoder   = encoder
        self._queue:    asyncio.Queue = asyncio.Queue(maxsize=8)
        self._task:     Optional[asyncio.Task] = None
        self._pts       = 0
        self._time_base = fractions.Fraction(1, 90000)   # RTP clock

    async def start_consuming(self):
        self._task = asyncio.create_task(self._consume())

    async def _consume(self):
        async for packet in self._encoder.packet_iterator():
            try:
                self._queue.put_nowait(packet)
            except asyncio.QueueFull:
                pass  # drop — client will get next keyframe

    async def recv(self):
        """Called by aiortc to get the next video frame."""
        data = await self._queue.get()
        packet = av.Packet(data)
        packet.pts      = self._pts
        packet.dts      = self._pts
        packet.time_base = self._time_base
        self._pts += int(90000 / self._encoder.fps)
        return packet

    async def stop(self):
        if self._task:
            self._task.cancel()
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
        self._encoder: Optional[SessionEncoder] = None
        self._watcher: Optional[MemoryWatcher]  = None
        self._clients: dict[str, ConnectedClient] = {}
        self._loop:    Optional[asyncio.AbstractEventLoop] = None

        # Core runs in a dedicated thread
        self._core_thread: Optional[threading.Thread] = None
        self._running = False

        # Input state: player_num → button bitmask
        self._inputs: dict[int, int] = {i: 0 for i in range(4)}
        self._input_lock = threading.Lock()

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

        # 2. Start encoder
        self._encoder = encode_manager.create(
            self.session_id,
            self._core.width,
            self._core.height,
            self._core.fps,
            self._loop,
        )

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

        # Close all peer connections
        for client in list(self._clients.values()):
            await client.pc.close()
        self._clients.clear()

        encode_manager.destroy(self.session_id)

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

        while self._running:
            # Apply merged inputs
            with self._input_lock:
                for player, buttons in self._inputs.items():
                    self._core.set_input(player, buttons)

            self._core.run()

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
        if self._encoder:
            self._encoder.push_frame(rgb24)

    def _on_audio(self, pcm: bytes):
        # TODO: feed into aiortc audio track (future: per-client audio mixing)
        pass

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

        # Video track
        video_track = LibretroVideoTrack(self._encoder)
        await video_track.start_consuming()
        pc.addTrack(video_track)

        # Handle ICE connection state
        @pc.on("connectionstatechange")
        async def on_state():
            log.info("[%s] Client %s: %s", self.session_id, client_id,
                     pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._remove_client(client_id)

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
                        # Add audio track to their PC if not already present
                        # (Full implementation would use a mixing bus)
                        pass
            except MediaStreamError:
                break

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def apply_input(self, client_id: str, buttons: int):
        """
        Apply a button bitmask from a client.
        Maps client → their assigned player slot.
        """
        client = self._clients.get(client_id)
        if client and client.player_num is not None:
            with self._input_lock:
                self._inputs[client.player_num] = buttons

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
