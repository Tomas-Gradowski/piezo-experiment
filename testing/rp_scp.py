"""
SCP/SSH helpers.
We use system scp/ssh (simple, no extra libs).
Assumes you can ssh into the RP (often root@rp-xxxx.local).
"""

import subprocess

def ssh_mkdir(user: str, host: str, remote_path: str):
    subprocess.check_call(["ssh", f"{user}@{host}", "mkdir", "-p", remote_path])

def scp_pull_dir(user: str, host: str, remote_path: str, local_path: str):
    # Pull entire directory recursively
    subprocess.check_call(["scp", "-r", f"{user}@{host}:{remote_path}", local_path])
