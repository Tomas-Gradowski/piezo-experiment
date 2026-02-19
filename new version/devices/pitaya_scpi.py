from __future__ import annotations

import socket
from typing import List, Optional


class scpi:
    """
    Minimal SCPI-over-TCP client for Red Pitaya.

    Contract used by executor:
    - open(): ensure TCP connection is established
    - close(): close TCP connection
    - run_commands([...]): send setup commands (no immediate response)
    - query(cmd): send command and return one text response
    """

    def __init__(self, host: str, port: int = 5000, timeout: float = 10.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout) if timeout is not None else None
        self._socket: Optional[socket.socket] = None

        # Keep compatibility with existing behavior: connect at construction.
        self.open()
        self.delimiter = "\r\n"

    def open(self) -> None:
        if self._socket is not None:
            return

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.timeout is not None:
            s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        self._socket = s

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("SCPI socket not connected")
        return self._socket

    def tx_txt(self, cmd: str) -> None:
        sock = self._require_socket()
        # Match Red Pitaya reference client delimiter.
        sock.sendall((cmd.strip() + self.delimiter).encode("utf-8"))

    def _readline(self) -> str:
        sock = self._require_socket()
        data = bytearray()
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)

            # Red Pitaya typically returns either "{...}" or line-terminated text.
            if b"}" in chunk or chunk.endswith(b"\n"):
                break

            # Safety cap against misbehaving endpoint.
            if len(data) > 50_000_000:
                raise RuntimeError("SCPI response too large (possible read hang)")

        return data.decode("utf-8", errors="replace").strip()

    def txrx_txt(self, cmd: str) -> str:
        self.tx_txt(cmd)
        return self._readline()

    def send(self, cmd: str) -> None:
        self.tx_txt(cmd)

    def query(self, cmd: str) -> str:
        return self.txrx_txt(cmd)

    def run_commands(self, commands: List[str]) -> None:
        for cmd in commands:
            self.tx_txt(cmd)
