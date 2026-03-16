"""
session_manager.py

Singleton that owns all active GameSessions. Provides create/get/destroy
and a status overview for the admin API.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from .session import GameSession, SessionConfig

log = logging.getLogger(__name__)


class SessionManager:

    def __init__(self):
        self._sessions: dict[str, GameSession] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(self, core_path: str, rom_path: str,
                              max_players: int = 4) -> GameSession:
        """
        Validate paths, create a GameSession, start it, and register it.
        Raises FileNotFoundError or RuntimeError on failure.
        """
        if not Path(core_path).exists():
            raise FileNotFoundError(f"Core not found: {core_path}")
        if not Path(rom_path).exists():
            raise FileNotFoundError(f"ROM not found: {rom_path}")

        config  = SessionConfig(
            core_path   = core_path,
            rom_path    = rom_path,
            max_players = max_players,
        )
        session = await GameSession.create(config)
        await session.start()

        self._sessions[session.session_id] = session
        log.info("Session created: %s", session.session_id)
        return session

    async def destroy_session(self, session_id: str):
        session = self._sessions.pop(session_id, None)
        if session:
            await session.stop()
            log.info("Session destroyed: %s", session_id)

    def get_session(self, session_id: str) -> Optional[GameSession]:
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        return [s.info() for s in self._sessions.values()]

    def status(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "sessions":        self.list_sessions(),
            "active_tracks":   len(self._sessions),
        }

    async def shutdown(self):
        """Gracefully stop all sessions."""
        log.info("Shutting down %d sessions...", len(self._sessions))
        await asyncio.gather(
            *[s.stop() for s in self._sessions.values()],
            return_exceptions=True
        )
        self._sessions.clear()


# Global singleton
session_manager = SessionManager()
