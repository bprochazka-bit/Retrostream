"""
core_downloader.py

Downloads and updates libretro cores from the official buildbot.

The buildbot index at https://buildbot.libretro.com/nightly/linux/x86_64/latest/
serves pre-built .so.zip archives.  This module:

  - Scrapes the index to list available cores
  - Downloads and extracts .so files into the local cores/ directory
  - Tracks installed core metadata (version timestamp) in cores/.manifest.json
  - Supports checking for updates by comparing remote Last-Modified headers
"""

import asyncio
import io
import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)

BUILDBOT_BASE = "https://buildbot.libretro.com/nightly/linux/x86_64/latest"
CORES_DIR = Path("cores")
MANIFEST_PATH = CORES_DIR / ".manifest.json"

# Timeout for individual HTTP requests
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=120)

# Headers that avoid Brotli (aiohttp may lack brotli support)
_HEADERS = {"Accept-Encoding": "gzip, deflate"}


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


class CoreDownloader:

    def __init__(self, cores_dir: Optional[Path] = None):
        self.cores_dir = cores_dir or CORES_DIR
        self.cores_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_remote_cores(self) -> list[dict]:
        """Scrape the buildbot index and return available cores."""
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT, headers=_HEADERS) as session:
            async with session.get(f"{BUILDBOT_BASE}/") as resp:
                resp.raise_for_status()
                html = await resp.text()

        # Parse href links that end in _libretro.so.zip
        pattern = re.compile(r'href="([^"]+_libretro\.so\.zip)"', re.IGNORECASE)
        cores = []
        for match in pattern.finditer(html):
            # href may be a full path like /nightly/linux/x86_64/latest/foo.so.zip
            filename = match.group(1).rsplit("/", 1)[-1]
            core_name = filename.replace("_libretro.so.zip", "")
            cores.append({
                "name": core_name,
                "filename": filename,
                "url": f"{BUILDBOT_BASE}/{filename}",
            })

        cores.sort(key=lambda c: c["name"])
        return cores

    async def list_installed_cores(self) -> list[dict]:
        """Return cores currently present in the cores directory."""
        manifest = _load_manifest()
        installed = []
        for so_file in sorted(self.cores_dir.glob("*_libretro.so")):
            core_name = so_file.stem.replace("_libretro", "")
            meta = manifest.get(so_file.name, {})
            installed.append({
                "name": core_name,
                "filename": so_file.name,
                "path": str(so_file),
                "size_bytes": so_file.stat().st_size,
                "installed_at": meta.get("installed_at"),
                "remote_date": meta.get("remote_date"),
            })
        return installed

    async def download_core(self, core_name: str) -> dict:
        """
        Download a core from the buildbot.
        core_name can be e.g. "snes9x" or "snes9x_libretro.so.zip".
        Returns info about the installed core.
        """
        # Normalise to zip filename
        if core_name.endswith(".so.zip"):
            zip_name = core_name
        elif core_name.endswith("_libretro"):
            zip_name = f"{core_name}.so.zip"
        else:
            zip_name = f"{core_name}_libretro.so.zip"

        url = f"{BUILDBOT_BASE}/{zip_name}"
        so_name = zip_name.removesuffix(".zip")
        dest = self.cores_dir / so_name

        log.info("Downloading core: %s", url)

        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT, headers=_HEADERS) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                remote_date = None
                if "Last-Modified" in resp.headers:
                    try:
                        dt = parsedate_to_datetime(resp.headers["Last-Modified"])
                        remote_date = dt.isoformat()
                    except Exception:
                        pass

        # Extract the .so from the zip
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Find the .so inside (usually just one file)
            so_entries = [n for n in zf.namelist() if n.endswith(".so")]
            if not so_entries:
                raise ValueError(f"No .so file found in {zip_name}")
            zf.extract(so_entries[0], path=str(self.cores_dir))
            # If extracted name differs from expected, rename
            extracted = self.cores_dir / so_entries[0]
            if extracted != dest:
                extracted.rename(dest)

        # Make executable
        dest.chmod(0o755)

        # Update manifest
        manifest = _load_manifest()
        manifest[so_name] = {
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "remote_date": remote_date,
            "source_url": url,
        }
        _save_manifest(manifest)

        log.info("Core installed: %s (%d bytes)", dest, dest.stat().st_size)
        return {
            "name": so_name.replace("_libretro.so", ""),
            "filename": so_name,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "installed_at": manifest[so_name]["installed_at"],
            "remote_date": remote_date,
        }

    async def check_updates(self) -> list[dict]:
        """
        Check installed cores against the buildbot.
        Returns a list of cores that have newer versions available.
        """
        manifest = _load_manifest()
        installed = {f.name for f in self.cores_dir.glob("*_libretro.so")}

        if not installed:
            return []

        updates = []
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT, headers=_HEADERS) as session:
            for so_name in sorted(installed):
                zip_name = f"{so_name}.zip"
                url = f"{BUILDBOT_BASE}/{zip_name}"
                meta = manifest.get(so_name, {})
                old_date = meta.get("remote_date")

                try:
                    async with session.head(url) as resp:
                        if resp.status != 200:
                            continue
                        if "Last-Modified" not in resp.headers:
                            continue
                        dt = parsedate_to_datetime(resp.headers["Last-Modified"])
                        new_date = dt.isoformat()
                except Exception:
                    continue

                if old_date is None or new_date != old_date:
                    updates.append({
                        "name": so_name.replace("_libretro.so", ""),
                        "filename": so_name,
                        "installed_date": old_date,
                        "remote_date": new_date,
                    })

        return updates

    async def delete_core(self, core_name: str) -> bool:
        """Remove an installed core and its manifest entry."""
        if core_name.endswith("_libretro.so"):
            so_name = core_name
        elif core_name.endswith("_libretro"):
            so_name = f"{core_name}.so"
        else:
            so_name = f"{core_name}_libretro.so"

        path = self.cores_dir / so_name
        if not path.exists():
            return False

        path.unlink()
        manifest = _load_manifest()
        manifest.pop(so_name, None)
        _save_manifest(manifest)
        log.info("Core deleted: %s", so_name)
        return True


# Global singleton
core_downloader = CoreDownloader()
