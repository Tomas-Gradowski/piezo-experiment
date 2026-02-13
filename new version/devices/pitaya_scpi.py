from __future__ import annotations
import socket
from typing import List



# -----------------------------
# SCPI client
# -----------------------------

class PitayaSCPI:
    """
    Minimal SCPI over TCP client for Red Pitaya.
    - send(cmd): writes a command (no response expected)
    - query(cmd): writes and reads until newline (response expected)
    """
    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        s.settimeout(self.timeout_s)
        self.sock = s

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def send(self, cmd: str) -> None:
        if not self.sock:
            raise RuntimeError("SCPI socket not connected")
        payload = (cmd.strip() + "\n").encode("utf-8")
        self.sock.sendall(payload)

    def query(self, cmd: str) -> str:
        self.send(cmd)
        return self._readline()

    def _readline(self) -> str:
        if not self.sock:
            raise RuntimeError("SCPI socket not connected")
        data = bytearray()
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
            if b"}" in chunk or chunk.endswith(b"\n"):
                break

            # safety cap (prevents infinite loop if device misbehaves)
            if len(data) > 50_000_000:
                raise RuntimeError("SCPI response too large (possible read hang)")

        return data.decode("utf-8", errors="replace").strip()


