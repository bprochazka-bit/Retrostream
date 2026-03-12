"""
memory_watcher.py

Polls the libretro core's system RAM on a timer, diffs named address
values from a YAML config, and pushes changes to subscribers over WebSocket.

Memory map YAML format (configs/memory_maps/<rom_name>.yml):
    lives:
      addr: 0x0DBE
      type: u8
    score:
      addr: 0x0F34
      type: u24
      endian: little
    world:
      addr: 0x0760
      type: u8
      bitmask: 0x0F    # optional: mask before comparing
"""

import asyncio
import logging
import struct
import time
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported value types
# ---------------------------------------------------------------------------

_FORMATS = {
    "u8":   (1, "B"),
    "s8":   (1, "b"),
    "u16":  (2, "H"),
    "s16":  (2, "h"),
    "u24":  (3, None),   # handled specially
    "u32":  (4, "I"),
    "s32":  (4, "i"),
}


def read_value(mem: bytes, addr: int, type_str: str,
               endian: str = "little", bitmask: Optional[int] = None) -> Optional[int]:
    """Read a typed value from a memory snapshot."""
    size, fmt = _FORMATS.get(type_str, (None, None))
    if size is None:
        return None
    if addr + size > len(mem):
        return None

    raw = mem[addr:addr + size]

    if type_str == "u24":
        if endian == "little":
            val = raw[0] | (raw[1] << 8) | (raw[2] << 16)
        else:
            val = (raw[0] << 16) | (raw[1] << 8) | raw[2]
    else:
        order = "<" if endian == "little" else ">"
        val = struct.unpack(f"{order}{fmt}", raw)[0]

    if bitmask is not None:
        val &= bitmask

    return val


# ---------------------------------------------------------------------------
# Memory map loader
# ---------------------------------------------------------------------------

class MemoryMap:
    """Parsed memory map from a YAML config file."""

    def __init__(self, config: dict):
        self.fields: dict[str, dict] = {}
        for name, spec in config.items():
            self.fields[name] = {
                "addr":    int(str(spec["addr"]), 0),   # handles 0x hex
                "type":    spec.get("type", "u8"),
                "endian":  spec.get("endian", "little"),
                "bitmask": spec.get("bitmask"),
            }

    @classmethod
    def from_file(cls, path: str) -> "MemoryMap":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data)

    @classmethod
    def empty(cls) -> "MemoryMap":
        return cls({})

    def read_all(self, mem: bytes) -> dict[str, Any]:
        """Read all fields from a memory snapshot. Returns {name: value}."""
        out = {}
        for name, spec in self.fields.items():
            val = read_value(
                mem, spec["addr"], spec["type"],
                spec["endian"], spec["bitmask"]
            )
            if val is not None:
                out[name] = val
        return out


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

class MemoryWatcher:
    """
    Polls the core's RAM at `poll_hz` and notifies subscribers
    when any named field changes value.

    Subscribers receive: {"type": "memory_update", "changes": {"lives": 3, ...}}
    """

    def __init__(self, session_id: str,
                 get_memory_fn: Callable[[], Optional[bytes]],
                 memory_map: MemoryMap,
                 poll_hz: float = 10.0):
        self.session_id   = session_id
        self._get_memory  = get_memory_fn
        self._map         = memory_map
        self._poll_hz     = poll_hz
        self._interval    = 1.0 / poll_hz
        self._last_values: dict[str, Any] = {}
        self._subscribers: list[asyncio.Queue] = []
        self._running     = False
        self._task: Optional[asyncio.Task] = None

    def subscribe(self) -> asyncio.Queue:
        """Returns a queue that will receive change dicts."""
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = [s for s in self._subscribers if s is not q]

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _poll_loop(self):
        log.info("[%s] Memory watcher started at %.1fHz with %d fields",
                 self.session_id, self._poll_hz, len(self._map.fields))
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                await self._tick()
            except Exception as e:
                log.debug("[%s] Memory poll error: %s", self.session_id, e)

    async def _tick(self):
        mem = self._get_memory()
        if not mem:
            return

        current = self._map.read_all(mem)
        changes = {
            k: v for k, v in current.items()
            if self._last_values.get(k) != v
        }

        if changes:
            self._last_values.update(changes)
            msg = {"type": "memory_update", "changes": changes,
                   "ts": time.time()}
            await self._broadcast(msg)

    async def _broadcast(self, msg: dict):
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

    def snapshot(self) -> dict[str, Any]:
        """Return the last known values of all fields (for new subscribers)."""
        return dict(self._last_values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_memory_map_for_rom(rom_path: str,
                             maps_dir: str = "configs/memory_maps") -> MemoryMap:
    """
    Try to find a memory map YAML for the given ROM.
    Matches by ROM filename (without extension).
    Falls back to an empty map if none found.
    """
    rom_name  = Path(rom_path).stem.lower()
    maps_path = Path(maps_dir)

    # Exact match first
    candidate = maps_path / f"{rom_name}.yml"
    if candidate.exists():
        log.info("Loaded memory map: %s", candidate)
        return MemoryMap.from_file(str(candidate))

    # Fuzzy: check if any map filename is a substring of the ROM name
    if maps_path.exists():
        for f in maps_path.glob("*.yml"):
            if f.stem.lower() in rom_name:
                log.info("Fuzzy-matched memory map: %s", f)
                return MemoryMap.from_file(str(f))

    log.info("No memory map found for '%s' — watcher will be passive", rom_name)
    return MemoryMap.empty()
