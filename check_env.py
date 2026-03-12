#!/usr/bin/env python3
"""
check_env.py

Verifies all RetroStream dependencies on a bare Debian 13 (trixie) system.
All packages are installed via apt — no pip required.

Usage:
    python3 check_env.py
"""

import subprocess
import sys
from pathlib import Path


def check(label: str, ok: bool, fix: str = "", warn_only: bool = False) -> bool:
    symbol = "✓" if ok else ("⚠" if warn_only else "✗")
    color  = "\033[32m" if ok else ("\033[33m" if warn_only else "\033[31m")
    reset  = "\033[0m"
    print(f"  {color}{symbol}{reset}  {label}", end="")
    if not ok and fix:
        print(f"\n       → {fix}", end="")
    print()
    return ok or warn_only


def run(cmd: list) -> tuple:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.returncode, r.stdout + r.stderr
    except FileNotFoundError:
        return -1, "not found"
    except Exception as e:
        return -1, str(e)


def main() -> int:
    print("\n=== RetroStream Environment Check (Debian 13 trixie) ===\n")
    results = []

    # ── Python ────────────────────────────────────────────────────────────────
    print("Python:")
    maj, min_ = sys.version_info[:2]
    results.append(check(
        f"python3 {maj}.{min_}",
        maj == 3 and min_ >= 11,
        fix="apt install python3   (trixie ships 3.13)"
    ))

    # ── Python packages (all from apt) ────────────────────────────────────────
    print("\nPython packages (apt):")
    apt_pkgs = {
        "fastapi":    "python3-fastapi",
        "uvicorn":    "python3-uvicorn",
        "aiortc":     "python3-aiortc",
        "websockets": "python3-websockets",
        "numpy":      "python3-numpy",
        "yaml":       "python3-yaml",
        "pyee":       "python3-pyee",
        "aiofiles":   "python3-aiofiles",
        "multipart":  "python3-multipart",
        "pydantic":   "python3-pydantic",
    }
    for module, apt_name in apt_pkgs.items():
        try:
            __import__(module)
            results.append(check(f"{module}  ({apt_name})", True))
        except ImportError:
            results.append(check(
                f"{module}  ({apt_name})", False,
                fix=f"apt install {apt_name}"
            ))

    try:
        import av  # noqa: F401
        results.append(check("av / PyAV  (python3-av)", True))
    except ImportError:
        results.append(check(
            "av / PyAV  (python3-av)", False,
            fix="apt install python3-av"
        ))

    # ── FFmpeg ────────────────────────────────────────────────────────────────
    print("\nFFmpeg:")
    rc, _ = run(["ffmpeg", "-version"])
    results.append(check("ffmpeg binary", rc == 0, fix="apt install ffmpeg"))

    rc, out = run(["ffmpeg", "-hide_banner", "-hwaccels"])
    check("FFmpeg VAAPI support", "vaapi" in out.lower(),
          fix="Encoder falls back to libx264 if unavailable", warn_only=True)
    check("FFmpeg QSV support", "qsv" in out.lower(), warn_only=True)

    # ── VAAPI / Intel GPU ─────────────────────────────────────────────────────
    print("\nIntel GPU / VAAPI:")
    dev = Path("/dev/dri/renderD128")
    check(
        "VAAPI device /dev/dri/renderD128", dev.exists(),
        fix="apt install intel-media-va-driver  — falls back to software if absent",
        warn_only=True
    )
    rc, out = run(["vainfo"])
    check("vainfo H264 encode profile", "H264" in out or "h264" in out.lower(),
          fix="apt install intel-media-va-driver  or  i965-va-driver", warn_only=True)
    rc, _ = run(["intel_gpu_top", "--help"])
    check("intel_gpu_top (GPU monitor)", rc != -1,
          fix="apt install intel-gpu-tools", warn_only=True)

    # ── Directories ───────────────────────────────────────────────────────────
    print("\nDirectories:")
    for d in ["cores", "roms", "configs/memory_maps", "saves", "system"]:
        exists = Path(d).exists()
        results.append(check(d, exists, fix=f"mkdir -p {d}"))

    # ── Cores and ROMs ────────────────────────────────────────────────────────
    print("\nContent:")
    cores = list(Path("cores").glob("*.so")) if Path("cores").exists() else []
    check(f"libretro cores ({len(cores)} found)", len(cores) > 0,
          fix="Place .so cores in ./cores/  (e.g. snes9x_libretro.so)", warn_only=True)
    for c in cores[:5]:
        print(f"       → {c.name}")

    roms = [f for f in Path("roms").iterdir()
            if f.is_file()] if Path("roms").exists() else []
    check(f"ROMs ({len(roms)} found)", len(roms) > 0,
          fix="Place ROM files in ./roms/", warn_only=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    hard_fails = [r for r in results if not r]
    print()
    if not hard_fails:
        print("\033[32m✓  All required checks passed.\033[0m")
        print("   Start with:  uvicorn server.main:app --host 0.0.0.0 --port 8000\n")
        return 0
    else:
        print(f"\033[31m✗  {len(hard_fails)} required check(s) failed. See above.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
