"""
libretro_core.py

ctypes wrapper around a libretro .so core.
Exposes frame callbacks, input injection, audio, and memory map access.

Usage:
    core = LibretroCore("cores/snes9x_libretro.so")
    core.load_game("roms/mario.sfc")
    core.start(frame_cb=my_frame_handler, audio_cb=my_audio_handler)
    while running:
        core.run()  # advances one frame
    core.unload()
"""

import ctypes
import ctypes.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# libretro constants
# ---------------------------------------------------------------------------
RETRO_ENVIRONMENT_SET_ROTATION                = 1
RETRO_ENVIRONMENT_GET_OVERSCAN                = 2
RETRO_ENVIRONMENT_GET_CAN_DUPE                = 3
RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY        = 9
RETRO_ENVIRONMENT_SET_PIXEL_FORMAT            = 10
RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS       = 11
RETRO_ENVIRONMENT_GET_VARIABLE                = 15
RETRO_ENVIRONMENT_SET_VARIABLES               = 16
RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE         = 17
RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME         = 18
RETRO_ENVIRONMENT_GET_LOG_INTERFACE           = 27
RETRO_ENVIRONMENT_SET_SUBSYSTEM_INFO          = 30
RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY          = 31
RETRO_ENVIRONMENT_SET_GEOMETRY                = 37
RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION    = 52
RETRO_ENVIRONMENT_SET_CORE_OPTIONS            = 53
RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2_INTL    = 68
RETRO_ENVIRONMENT_SET_HW_RENDER               = 14
RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE        = 35
RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE     = 65

RETRO_PIXEL_FORMAT_0RGB1555 = 0
RETRO_PIXEL_FORMAT_XRGB8888 = 1
RETRO_PIXEL_FORMAT_RGB565   = 2

RETRO_MEMORY_SAVE_RAM   = 0
RETRO_MEMORY_RTC        = 1
RETRO_MEMORY_SYSTEM_RAM = 2
RETRO_MEMORY_VIDEO_RAM  = 3

RETRO_DEVICE_JOYPAD  = 1
RETRO_DEVICE_ANALOG  = 5

# Joypad button indices
RETRO_DEVICE_ID_JOYPAD_B      = 0
RETRO_DEVICE_ID_JOYPAD_Y      = 1
RETRO_DEVICE_ID_JOYPAD_SELECT = 2
RETRO_DEVICE_ID_JOYPAD_START  = 3
RETRO_DEVICE_ID_JOYPAD_UP     = 4
RETRO_DEVICE_ID_JOYPAD_DOWN   = 5
RETRO_DEVICE_ID_JOYPAD_LEFT   = 6
RETRO_DEVICE_ID_JOYPAD_RIGHT  = 7
RETRO_DEVICE_ID_JOYPAD_A      = 8
RETRO_DEVICE_ID_JOYPAD_X      = 9
RETRO_DEVICE_ID_JOYPAD_L      = 10
RETRO_DEVICE_ID_JOYPAD_R      = 11

# ---------------------------------------------------------------------------
# C struct definitions
# ---------------------------------------------------------------------------
class RetroGameInfo(ctypes.Structure):
    _fields_ = [
        ("path",  ctypes.c_char_p),
        ("data",  ctypes.c_void_p),
        ("size",  ctypes.c_size_t),
        ("meta",  ctypes.c_char_p),
    ]

class RetroSystemInfo(ctypes.Structure):
    _fields_ = [
        ("library_name",     ctypes.c_char_p),
        ("library_version",  ctypes.c_char_p),
        ("valid_extensions", ctypes.c_char_p),
        ("need_fullpath",    ctypes.c_bool),
        ("block_extract",    ctypes.c_bool),
    ]

class RetroGameGeometry(ctypes.Structure):
    _fields_ = [
        ("base_width",   ctypes.c_uint),
        ("base_height",  ctypes.c_uint),
        ("max_width",    ctypes.c_uint),
        ("max_height",   ctypes.c_uint),
        ("aspect_ratio", ctypes.c_float),
    ]

class RetroSystemTiming(ctypes.Structure):
    _fields_ = [
        ("fps",         ctypes.c_double),
        ("sample_rate", ctypes.c_double),
    ]

class RetroAvInfo(ctypes.Structure):
    _fields_ = [
        ("geometry", RetroGameGeometry),
        ("timing",   RetroSystemTiming),
    ]

class RetroVariable(ctypes.Structure):
    _fields_ = [
        ("key",   ctypes.c_char_p),
        ("value", ctypes.c_char_p),
    ]

# ---------------------------------------------------------------------------
# Callback typedefs
# ---------------------------------------------------------------------------
EnvCallbackType    = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
VideoCallbackType  = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_size_t)
AudioSampleType    = ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int16)
AudioCallbackType  = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t)
InputPollType      = ctypes.CFUNCTYPE(None)
InputStateType     = ctypes.CFUNCTYPE(ctypes.c_int16, ctypes.c_uint, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_uint)

# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------
@dataclass
class InputState:
    """Holds button bitmask for up to 4 players."""
    buttons: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def set_button(self, player: int, button_id: int, pressed: bool):
        if 0 <= player < 4:
            if pressed:
                self.buttons[player] |= (1 << button_id)
            else:
                self.buttons[player] &= ~(1 << button_id)

    def get_button(self, player: int, button_id: int) -> int:
        if 0 <= player < 4:
            return 1 if (self.buttons[player] & (1 << button_id)) else 0
        return 0


class LibretroCore:
    """
    Wraps a libretro .so and exposes a simple Python interface.
    All callbacks are kept as instance attributes so they aren't GC'd.
    """

    def __init__(self, core_path: str, system_dir: str = "./system",
                 save_dir: str = "./saves"):
        self.core_path  = Path(core_path)
        self.system_dir = system_dir.encode()
        self.save_dir   = save_dir.encode()

        self._lib      = ctypes.CDLL(str(self.core_path))
        self._input    = InputState()
        self._pixel_fmt = RETRO_PIXEL_FORMAT_XRGB8888

        # User-supplied callbacks
        self.on_frame: Optional[Callable[[bytes, int, int], None]] = None
        self.on_audio: Optional[Callable[[bytes], None]]           = None

        self._av_info: Optional[RetroAvInfo] = None

        # Prevent GC of buffers the core holds pointers to
        self._rom_buf  = None
        self._rom_path_buf = None
        self._game_info = None
        self._bind_functions()
        self._install_callbacks()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _bind_functions(self):
        lib = self._lib
        lib.retro_init.restype          = None
        lib.retro_deinit.restype        = None
        lib.retro_api_version.restype   = ctypes.c_uint
        lib.retro_get_system_info.restype   = None
        lib.retro_get_system_av_info.restype = None
        lib.retro_set_environment.restype   = None
        lib.retro_set_video_refresh.restype = None
        lib.retro_set_audio_sample.restype       = None
        lib.retro_set_audio_sample_batch.restype = None
        lib.retro_set_input_poll.restype    = None
        lib.retro_set_input_state.restype   = None
        lib.retro_load_game.restype         = ctypes.c_bool
        lib.retro_unload_game.restype       = None
        lib.retro_run.restype               = None
        lib.retro_reset.restype             = None
        lib.retro_get_memory_data.restype   = ctypes.c_void_p
        lib.retro_get_memory_size.restype   = ctypes.c_size_t
        lib.retro_serialize_size.restype    = ctypes.c_size_t
        lib.retro_serialize.restype         = ctypes.c_bool
        lib.retro_unserialize.restype       = ctypes.c_bool

    def _install_callbacks(self):
        # Keep references so CPython doesn't GC the cfunctype wrappers
        self._cb_env          = EnvCallbackType(self._env_callback)
        self._cb_video        = VideoCallbackType(self._video_callback)
        self._cb_audio_sample = AudioSampleType(self._audio_sample_callback)
        self._cb_audio        = AudioCallbackType(self._audio_callback)
        self._cb_poll         = InputPollType(self._input_poll)
        self._cb_state        = InputStateType(self._input_state)

        log.info("Installing callbacks...")
        log.info("  retro_set_environment...")
        self._lib.retro_set_environment(self._cb_env)
        log.info("  retro_set_environment OK")
        log.info("  retro_set_video_refresh...")
        self._lib.retro_set_video_refresh(self._cb_video)
        log.info("  retro_set_video_refresh OK")
        log.info("  retro_set_audio_sample...")
        self._lib.retro_set_audio_sample(self._cb_audio_sample)
        log.info("  retro_set_audio_sample OK")
        log.info("  retro_set_audio_sample_batch...")
        self._lib.retro_set_audio_sample_batch(self._cb_audio)
        log.info("  retro_set_audio_sample_batch OK")
        log.info("  retro_set_input_poll...")
        self._lib.retro_set_input_poll(self._cb_poll)
        log.info("  retro_set_input_poll OK")
        log.info("  retro_set_input_state...")
        self._lib.retro_set_input_state(self._cb_state)
        log.info("  All callbacks installed OK")

    # ------------------------------------------------------------------
    # libretro callbacks (called from C)
    # ------------------------------------------------------------------
    def _env_callback(self, cmd: int, data: ctypes.c_void_p) -> bool:
        try:
            return self._env_callback_inner(cmd, data)
        except Exception as e:
            log.error("ENV exception on cmd=%d (0x%x): %s", cmd, cmd, e)
            return False

    def _env_callback_inner(self, cmd: int, data: ctypes.c_void_p) -> bool:
        if cmd == RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
            fmt = ctypes.cast(data, ctypes.POINTER(ctypes.c_int)).contents.value
            self._pixel_fmt = fmt
            log.info("ENV SET_PIXEL_FORMAT: %d (%s)", fmt,
                     {0: "0RGB1555", 1: "XRGB8888", 2: "RGB565"}.get(fmt, "unknown"))
            return True

        if cmd == RETRO_ENVIRONMENT_GET_CAN_DUPE:
            # Tell core we support frame duplication (NULL video data = repeat last frame)
            ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))
            ptr[0] = True
            log.info("ENV GET_CAN_DUPE -> true")
            return True

        if cmd == RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
            ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))
            ptr[0] = self.system_dir
            log.info("ENV GET_SYSTEM_DIRECTORY -> %s", self.system_dir)
            return True

        if cmd == RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))
            ptr[0] = self.save_dir
            log.info("ENV GET_SAVE_DIRECTORY -> %s", self.save_dir)
            return True

        if cmd == RETRO_ENVIRONMENT_GET_VARIABLE:
            # Core is requesting a variable value — return NULL (no override)
            if data:
                var = ctypes.cast(data, ctypes.POINTER(RetroVariable))
                key = var[0].key
                log.info("ENV GET_VARIABLE: key=%s -> None", key)
                var[0].value = None
            return True

        if cmd == RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
            # "Have variables been updated?" — no
            if data:
                ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))
                ptr[0] = False
            return True

        if cmd == RETRO_ENVIRONMENT_SET_VARIABLES:
            log.info("ENV SET_VARIABLES: acknowledged")
            return True

        if cmd == RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME:
            log.info("ENV SET_SUPPORT_NO_GAME: acknowledged")
            return True

        if cmd == RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS:
            log.info("ENV SET_INPUT_DESCRIPTORS: acknowledged")
            return True

        if cmd == RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
            # Tell core we support options v0 (basic)
            ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))
            ptr[0] = 0
            log.info("ENV GET_CORE_OPTIONS_VERSION -> 0")
            return True

        if cmd == RETRO_ENVIRONMENT_SET_CORE_OPTIONS:
            log.info("ENV SET_CORE_OPTIONS: acknowledged")
            return True

        if cmd == RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2_INTL:
            log.info("ENV SET_CORE_OPTIONS_V2_INTL: acknowledged")
            return True

        if cmd == RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
            log.info("ENV GET_LOG_INTERFACE -> declined")
            return False

        if cmd == RETRO_ENVIRONMENT_SET_GEOMETRY:
            log.info("ENV SET_GEOMETRY: acknowledged")
            return True

        if cmd == RETRO_ENVIRONMENT_GET_OVERSCAN:
            if data:
                ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))
                ptr[0] = False
            log.info("ENV GET_OVERSCAN -> false")
            return True

        if cmd == RETRO_ENVIRONMENT_SET_HW_RENDER:
            log.warning("ENV SET_HW_RENDER -> declined (software rendering only)")
            return False

        if cmd == RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE:
            log.info("ENV GET_RUMBLE_INTERFACE -> declined")
            return False

        if cmd == RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE:
            log.warning("ENV GET_HW_RENDER_INTERFACE -> declined (no GPU context)")
            return False

        log.info("ENV unhandled cmd=%d (0x%x), data=%s", cmd, cmd, data)
        return False

    def _video_callback(self, data: ctypes.c_void_p, width: int,
                        height: int, pitch: int):
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count <= 3:
            log.info("VIDEO frame #%d: data=%s width=%d height=%d pitch=%d fmt=%d",
                     self._frame_count, data, width, height, pitch, self._pixel_fmt)
        if data and self.on_frame:
            # Convert to raw RGB24 bytes regardless of source pixel format
            raw = self._pixels_to_rgb24(data, width, height, pitch)
            self.on_frame(raw, width, height)

    def _audio_sample_callback(self, left: int, right: int):
        pass  # single-sample callback — most cores use the batch version

    def _audio_callback(self, data: ctypes.POINTER(ctypes.c_int16),
                        frames: int) -> int:
        if self.on_audio:
            size  = frames * 2 * ctypes.sizeof(ctypes.c_int16)
            raw   = ctypes.string_at(data, size)
            self.on_audio(raw)
        return frames

    def _input_poll(self):
        pass  # inputs are pre-loaded into self._input

    def _input_state(self, port: int, device: int,
                     index: int, button_id: int) -> int:
        if device == RETRO_DEVICE_JOYPAD:
            return self._input.get_button(port, button_id)
        return 0

    # ------------------------------------------------------------------
    # Pixel format conversion
    # ------------------------------------------------------------------
    def _pixels_to_rgb24(self, data: ctypes.c_void_p, width: int,
                          height: int, pitch: int) -> bytes:
        import numpy as np

        if self._pixel_fmt == RETRO_PIXEL_FORMAT_XRGB8888:
            arr = np.frombuffer(
                ctypes.string_at(data, pitch * height), dtype=np.uint32
            ).reshape(height, pitch // 4)[:, :width]
            r = ((arr >> 16) & 0xFF).astype(np.uint8)
            g = ((arr >>  8) & 0xFF).astype(np.uint8)
            b = ( arr        & 0xFF).astype(np.uint8)
            return np.stack([r, g, b], axis=2).tobytes()

        elif self._pixel_fmt == RETRO_PIXEL_FORMAT_RGB565:
            arr = np.frombuffer(
                ctypes.string_at(data, pitch * height), dtype=np.uint16
            ).reshape(height, pitch // 2)[:, :width]
            r = ((arr >> 11) & 0x1F).astype(np.uint8) * 8
            g = ((arr >>  5) & 0x3F).astype(np.uint8) * 4
            b = ( arr        & 0x1F).astype(np.uint8) * 8
            return np.stack([r, g, b], axis=2).tobytes()

        else:  # 0RGB1555
            arr = np.frombuffer(
                ctypes.string_at(data, pitch * height), dtype=np.uint16
            ).reshape(height, pitch // 2)[:, :width]
            r = ((arr >> 10) & 0x1F).astype(np.uint8) * 8
            g = ((arr >>  5) & 0x1F).astype(np.uint8) * 8
            b = ( arr        & 0x1F).astype(np.uint8) * 8
            return np.stack([r, g, b], axis=2).tobytes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init(self):
        # Query system info before init
        sys_info = RetroSystemInfo()
        self._lib.retro_get_system_info(ctypes.byref(sys_info))
        log.info("Core system info: name=%s version=%s extensions=%s need_fullpath=%s",
                 sys_info.library_name, sys_info.library_version,
                 sys_info.valid_extensions, sys_info.need_fullpath)
        self._need_fullpath = bool(sys_info.need_fullpath)

        log.info("Calling retro_init()...")
        self._lib.retro_init()
        log.info("retro_init() complete: %s (API version %d)",
                 self.core_path.name, self._lib.retro_api_version())

    def load_game(self, rom_path: str) -> bool:
        rom_abs = str(Path(rom_path).resolve())
        self._rom_path_buf = rom_abs.encode()
        log.info("load_game: rom_path=%s (resolved=%s)", rom_path, rom_abs)
        log.info("load_game: need_fullpath=%s", self._need_fullpath)

        if self._need_fullpath:
            # Core reads the file itself — don't pass data, just the path
            self._rom_buf = None
            self._game_info       = RetroGameInfo()
            self._game_info.path  = self._rom_path_buf
            self._game_info.data  = None
            self._game_info.size  = 0
            self._game_info.meta  = None
            log.info("load_game: fullpath mode — passing path only")
        else:
            rom_data = Path(rom_path).read_bytes()
            self._rom_buf = ctypes.create_string_buffer(rom_data)
            self._game_info       = RetroGameInfo()
            self._game_info.path  = self._rom_path_buf
            self._game_info.data  = ctypes.cast(self._rom_buf, ctypes.c_void_p)
            self._game_info.size  = len(rom_data)
            self._game_info.meta  = None
            log.info("load_game: buffered mode — %d bytes loaded into memory", len(rom_data))

        log.info("load_game: calling retro_load_game()...")
        ok = self._lib.retro_load_game(ctypes.byref(self._game_info))
        log.info("load_game: retro_load_game() returned %s", ok)

        if ok:
            self._av_info = RetroAvInfo()
            self._lib.retro_get_system_av_info(ctypes.byref(self._av_info))
            log.info("ROM loaded: %dx%d @ %.2fHz, sample_rate=%.0f",
                     self._av_info.geometry.base_width,
                     self._av_info.geometry.base_height,
                     self._av_info.timing.fps,
                     self._av_info.timing.sample_rate)
            log.info("ROM geometry: max=%dx%d aspect=%.4f",
                     self._av_info.geometry.max_width,
                     self._av_info.geometry.max_height,
                     self._av_info.geometry.aspect_ratio)
        else:
            log.error("Failed to load ROM: %s", rom_path)
        return ok

    def run(self):
        """Advance one frame. Triggers video/audio callbacks."""
        self._lib.retro_run()

    def reset(self):
        self._lib.retro_reset()

    def unload(self):
        self._lib.retro_unload_game()
        self._lib.retro_deinit()

    def set_input(self, player: int, buttons: int):
        """Set the full button bitmask for a player (0-indexed)."""
        if 0 <= player < 4:
            self._input.buttons[player] = buttons

    @property
    def fps(self) -> float:
        return self._av_info.timing.fps if self._av_info else 60.0

    @property
    def width(self) -> int:
        return self._av_info.geometry.base_width if self._av_info else 256

    @property
    def height(self) -> int:
        return self._av_info.geometry.base_height if self._av_info else 224

    @property
    def sample_rate(self) -> float:
        return self._av_info.timing.sample_rate if self._av_info else 44100.0

    def get_memory(self, memory_id: int = RETRO_MEMORY_SYSTEM_RAM) -> Optional[bytes]:
        """Return a snapshot of the requested memory region."""
        size = self._lib.retro_get_memory_size(memory_id)
        ptr  = self._lib.retro_get_memory_data(memory_id)
        if ptr and size:
            return ctypes.string_at(ptr, size)
        return None

    def get_memory_ptr(self, memory_id: int = RETRO_MEMORY_SYSTEM_RAM):
        """Return (ptr, size) for zero-copy polling."""
        size = self._lib.retro_get_memory_size(memory_id)
        ptr  = self._lib.retro_get_memory_data(memory_id)
        return ptr, size

    def save_state(self) -> Optional[bytes]:
        size = self._lib.retro_serialize_size()
        buf  = ctypes.create_string_buffer(size)
        if self._lib.retro_serialize(buf, size):
            return bytes(buf)
        return None

    def load_state(self, data: bytes) -> bool:
        buf = ctypes.create_string_buffer(data)
        return bool(self._lib.retro_unserialize(buf, len(data)))
