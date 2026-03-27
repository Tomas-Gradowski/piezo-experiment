from __future__ import annotations

import os
import socket
import time
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
        self.delimiter = "\r\n"
        self._connect_attempts = max(1, int(os.getenv("PITAYA_CONNECT_ATTEMPTS", "3")))
        self._retry_delay_s = max(0.0, float(os.getenv("PITAYA_CONNECT_RETRY_DELAY_S", "0.35")))

        # Optional comma-separated fallback hosts/IPs, e.g.:
        # PITAYA_HOST_FALLBACKS=rp-f06549.local,192.168.1.100
        self._fallback_hosts = [
            h.strip()
            for h in os.getenv("PITAYA_HOST_FALLBACKS", "169.254.93.42,rp-f06549.local").split(",")
            if h.strip()
        ]

        # Keep compatibility with existing behavior: connect at construction.
        self.open()

    def _connect_targets(self) -> List[str]:
        seen = set()
        targets: List[str] = []
        for h in [self.host, *self._fallback_hosts]:
            if h and h not in seen:
                seen.add(h)
                targets.append(h)
        return targets

    def open(self) -> None:
        if self._socket is not None:
            return

        targets = self._connect_targets()
        errors: List[str] = []
        per_try_timeout = self.timeout if self.timeout is not None else 3.0
        per_try_timeout = max(0.2, min(per_try_timeout, 2.0))

        for attempt in range(1, self._connect_attempts + 1):
            for tgt in targets:
                try:
                    s = socket.create_connection((tgt, self.port), timeout=per_try_timeout)
                    if self.timeout is not None:
                        s.settimeout(self.timeout)
                    self._socket = s
                    # Pin to the working host to accelerate subsequent reconnects.
                    self.host = tgt
                    return
                except OSError as e:
                    errors.append(f"{tgt}:{self.port} ({e})")
            if attempt < self._connect_attempts:
                time.sleep(self._retry_delay_s)

        msg = "; ".join(errors[-6:]) if errors else "no targets tried"
        raise RuntimeError(f"Failed to connect to Red Pitaya after {self._connect_attempts} attempts. {msg}")

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
        payload = (cmd.strip() + self.delimiter).encode("utf-8")
        for attempt in range(2):
            try:
                sock = self._require_socket()
                # Match Red Pitaya reference client delimiter.
                sock.sendall(payload)
                return
            except OSError:
                self.close()
                if attempt == 0:
                    self.open()
                else:
                    raise

    def _readline(self) -> str:
        for attempt in range(2):
            try:
                sock = self._require_socket()
                data = bytearray()
                saw_braced_payload = False
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data.extend(chunk)

                    # Red Pitaya typically returns either "{...}" or line-terminated text.
                    if b"}" in data:
                        if b"{" in data:
                            start = data.index(b"{")
                            end = data.index(b"}", start)
                            if end != -1:
                                saw_braced_payload = True
                                data = data[start:end + 1]
                                break
                        else:
                            # We saw a closing brace before any opening brace.
                            # Discard up to the last '}' and keep reading to resync.
                            cut = data.rfind(b"}")
                            if cut != -1:
                                data = data[cut + 1:]
                            continue
                    if chunk.endswith(b"\n") and not (b"{" in data and b"}" not in data):
                        break

                    # Safety cap against misbehaving endpoint.
                    if len(data) > 50_000_000:
                        raise RuntimeError("SCPI response too large (possible read hang)")

                text = data.decode("utf-8", errors="replace").strip()
                if saw_braced_payload and ("{" not in text or "}" not in text):
                    raise RuntimeError("Incomplete braced SCPI payload")
                return text
            except OSError:
                self.close()
                if attempt == 0:
                    self.open()
                else:
                    raise
        return ""

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
