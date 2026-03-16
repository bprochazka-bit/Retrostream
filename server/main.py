"""
main.py

FastAPI application entry point.

Routes:
    GET  /                          Serve frontend
    GET  /api/sessions              List active sessions
    POST /api/sessions              Create a new session
    GET  /api/sessions/{sid}        Session info
    DELETE /api/sessions/{sid}      Destroy session
    GET  /api/status                Overall server status

    GET    /api/cores/remote        List downloadable cores from buildbot
    GET    /api/cores/installed     List locally installed cores
    POST   /api/cores/download      Download a core
    GET    /api/cores/updates       Check for core updates
    DELETE /api/cores/{name}        Delete an installed core

    GET  /api/roms                  List available ROMs

    GET  /admin                     Admin dashboard

    POST /rtc/offer/{sid}           WebRTC SDP offer → answer

    WS   /ws/input/{sid}            Input WebSocket (per client)
    WS   /ws/memory/{sid}           Memory watch WebSocket (per client)
    WS   /ws/chat/{sid}             Text chat WebSocket
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiohttp

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session_manager import session_manager
from .core_downloader import core_downloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("RetroStream server starting")
    yield
    log.info("RetroStream server shutting down")
    await session_manager.shutdown()


app = FastAPI(title="RetroStream", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if present
frontend_path = Path(__file__).parent.parent / "frontend"
if (frontend_path / "dist").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "dist")), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    core_path:   str
    rom_path:    str
    max_players: int = 4

class RTCOfferRequest(BaseModel):
    sdp:       str
    type:      str
    client_id: Optional[str] = None
    role:      str = "player"   # "player" | "spectator"


# ---------------------------------------------------------------------------
# REST — sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    return session_manager.list_sessions()


@app.post("/api/sessions", status_code=201)
async def create_session(req: CreateSessionRequest):
    try:
        session = await session_manager.create_session(
            req.core_path, req.rom_path, req.max_players
        )
        return session.info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.info()


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await session_manager.destroy_session(session_id)


@app.get("/api/status")
async def server_status():
    return session_manager.status()


# ---------------------------------------------------------------------------
# REST — core management
# ---------------------------------------------------------------------------

class DownloadCoreRequest(BaseModel):
    core_name: str


@app.get("/api/cores/remote")
async def list_remote_cores():
    try:
        return await core_downloader.list_remote_cores()
    except Exception as e:
        log.exception("Failed to fetch remote core list")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/cores/installed")
async def list_installed_cores():
    return await core_downloader.list_installed_cores()


@app.post("/api/cores/download", status_code=201)
async def download_core(req: DownloadCoreRequest):
    try:
        result = await core_downloader.download_core(req.core_name)
        return result
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            raise HTTPException(status_code=404, detail=f"Core not found: {req.core_name}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        log.exception("Failed to download core")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cores/updates")
async def check_core_updates():
    try:
        return await core_downloader.check_updates()
    except Exception as e:
        log.exception("Failed to check for updates")
        raise HTTPException(status_code=502, detail=str(e))


@app.delete("/api/cores/{core_name}", status_code=204)
async def delete_core(core_name: str):
    deleted = await core_downloader.delete_core(core_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Core not found: {core_name}")


# ---------------------------------------------------------------------------
# REST — ROM listing
# ---------------------------------------------------------------------------

ROM_EXTENSIONS = {
    ".nes", ".sfc", ".smc", ".gb", ".gbc", ".gba", ".n64", ".z64", ".v64",
    ".gen", ".md", ".smd", ".gg", ".sms", ".pce", ".ngp", ".ngc", ".ws",
    ".wsc", ".a26", ".a78", ".lnx", ".jag", ".vb", ".nds", ".3ds",
    ".iso", ".bin", ".cue", ".chd", ".pbp", ".cso",
    ".zip", ".7z",
}

roms_dir = Path("roms")


@app.get("/api/roms")
async def list_roms():
    if not roms_dir.exists():
        return []
    roms = []
    for f in sorted(roms_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in ROM_EXTENSIONS:
            roms.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "extension": f.suffix.lower(),
            })
    return roms


# ---------------------------------------------------------------------------
# WebRTC signaling
# ---------------------------------------------------------------------------

@app.post("/rtc/offer/{session_id}")
async def rtc_offer(session_id: str, req: RTCOfferRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    client_id = req.client_id or str(uuid.uuid4())[:8]

    try:
        answer = await session.handle_offer(
            sdp       = req.sdp,
            sdp_type  = req.type,
            client_id = client_id,
            role      = req.role,
        )
        return answer
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception("WebRTC offer failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# WebSocket — input
# ---------------------------------------------------------------------------

@app.websocket("/ws/input/{session_id}")
async def ws_input(websocket: WebSocket, session_id: str):
    """
    Receives input messages from a client and injects into the session.

    Message format (JSON):
        {"client_id": "abc123", "buttons": 512}

    buttons is a bitmask using RETRO_DEVICE_ID_JOYPAD_* bit positions.
    The client_id must match one obtained from /rtc/offer.

    Keyboard-to-button mapping is done client-side; this endpoint
    only receives the resolved bitmask.
    """
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    log.info("Input WS connected: session=%s", session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg       = json.loads(raw)
                client_id = msg.get("client_id", "")
                buttons   = int(msg.get("buttons", 0))
                session.apply_input(client_id, buttons)
            except (json.JSONDecodeError, ValueError) as e:
                log.debug("Bad input message: %s", e)
    except WebSocketDisconnect:
        log.info("Input WS disconnected: session=%s", session_id)


# ---------------------------------------------------------------------------
# WebSocket — memory watch
# ---------------------------------------------------------------------------

@app.websocket("/ws/memory/{session_id}")
async def ws_memory(websocket: WebSocket, session_id: str):
    """
    Pushes memory change events to the client.

    On connect, sends a full snapshot of all known values immediately.
    Subsequent messages are diffs only:

        {"type": "memory_snapshot", "values": {"lives": 3, "score": 1200}}
        {"type": "memory_update",   "changes": {"lives": 2}, "ts": 1234567890.1}
    """
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    q = session.subscribe_memory()
    log.info("Memory WS connected: session=%s", session_id)

    # Send initial snapshot
    snapshot = session.memory_snapshot()
    if snapshot:
        await websocket.send_text(json.dumps({
            "type":   "memory_snapshot",
            "values": snapshot,
        }))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_text(json.dumps(msg))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        log.info("Memory WS disconnected: session=%s", session_id)
    finally:
        session.unsubscribe_memory(q)


# ---------------------------------------------------------------------------
# WebSocket — text chat
# ---------------------------------------------------------------------------

_chat_rooms: dict[str, list[WebSocket]] = {}


@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """
    Simple broadcast text chat for a session.

    Messages:
        Client → Server: {"username": "Player1", "text": "GG"}
        Server → Client: {"type": "chat", "username": "Player1", "text": "GG", "ts": ...}
    """
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    room = _chat_rooms.setdefault(session_id, [])
    room.append(websocket)
    log.info("Chat WS connected: session=%s (room size: %d)", session_id, len(room))

    try:
        import time as _time
        while True:
            raw = await websocket.receive_text()
            try:
                msg       = json.loads(raw)
                broadcast = json.dumps({
                    "type":     "chat",
                    "username": msg.get("username", "anonymous"),
                    "text":     str(msg.get("text", ""))[:500],
                    "ts":       _time.time(),
                })
                dead = []
                for ws in room:
                    try:
                        await ws.send_text(broadcast)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    room.remove(ws)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        if websocket in room:
            room.remove(websocket)
        log.info("Chat WS disconnected: session=%s (room size: %d)", session_id, len(room))


# ---------------------------------------------------------------------------
# Frontend fallback
# ---------------------------------------------------------------------------

@app.get("/admin")
async def admin_dashboard():
    admin_file = frontend_path / "admin.html"
    if admin_file.exists():
        return HTMLResponse(admin_file.read_text())
    raise HTTPException(status_code=404, detail="Admin page not found")


@app.get("/")
async def root():
    index = frontend_path / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("""
    <html><body>
    <h1>RetroStream</h1>
    <p>Backend running.</p>
    <ul>
      <li><a href="/docs">API Docs (Swagger)</a></li>
      <li><a href="/api/status">Server Status</a></li>
      <li><a href="/admin">Admin Dashboard</a></li>
    </ul>
    </body></html>
    """)
