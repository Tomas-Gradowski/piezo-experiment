"""
Red Pitaya SCPI client (TCP).
This will mirror what the C# code does with TcpClient + sendCommand + ReadData.

We will keep it minimal and robust:
- connect()
- send(cmd)
- query(cmd) -> str
- later: helper methods for acquisition
"""

import socket

class RedPitayaSCPI:
    def __init__(self, host: str, port: int = 5000, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send(self, cmd: str):
        assert self.sock is not None
        self.sock.sendall((cmd + "\r\n").encode("ascii"))

    def query(self, cmd: str) -> str:
        """
        Send command and read response.
        TODO: implement robust read (RP responses often end with }\r\n for DATA?).
        """
        self.send(cmd)
        # placeholder read:
        data = self.sock.recv(4096)
        return data.decode("ascii", errors="replace")
