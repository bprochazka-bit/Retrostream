#!/usr/bin/env python3
"""
Quick test to load a libretro core and run a few frames.
Usage:
    python3 test_core.py cores/mesen_libretro.so roms/tecmo_bowl.nes
"""
import sys
import logging
import signal

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("test_core")

# Catch segfaults
def segfault_handler(sig, frame):
    log.critical("SEGFAULT (signal %d) caught!", sig)
    sys.exit(139)

signal.signal(signal.SIGSEGV, segfault_handler)

from server.libretro_core import LibretroCore

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <core.so> <rom>")
        sys.exit(1)

    core_path = sys.argv[1]
    rom_path  = sys.argv[2]

    frame_count = 0

    def on_frame(rgb24, w, h):
        nonlocal frame_count
        frame_count += 1
        log.info("Frame %d: %dx%d (%d bytes)", frame_count, w, h, len(rgb24))

    def on_audio(pcm):
        pass

    log.info("=== Creating LibretroCore(%s) ===", core_path)
    core = LibretroCore(core_path)
    log.info("=== Core object created ===")

    core.on_frame = on_frame
    core.on_audio = on_audio

    log.info("=== Calling core.init() ===")
    core.init()
    log.info("=== init() complete ===")

    log.info("=== Calling core.load_game(%s) ===", rom_path)
    ok = core.load_game(rom_path)
    if not ok:
        log.error("load_game returned False!")
        sys.exit(1)
    log.info("=== load_game() complete ===")

    log.info("=== Running 10 frames ===")
    for i in range(10):
        log.info("--- retro_run() frame %d ---", i + 1)
        core.run()
        log.info("--- frame %d done ---", i + 1)

    log.info("=== Unloading ===")
    core.unload()
    log.info("=== Done! %d frames rendered ===", frame_count)

if __name__ == "__main__":
    main()
