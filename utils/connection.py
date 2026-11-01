"""Shared helpers for reaching a CARLA server from WSL."""
from __future__ import annotations

import os
import subprocess

import carla


def get_windows_host_ip() -> str:
    """WSL default gateway, or CARLA_HOST if set."""
    env = os.environ.get("CARLA_HOST")
    if env:
        return env
    result = subprocess.run(
        ["ip", "route", "show", "default"],
        capture_output=True,
        text=True,
        check=False,
    )
    parts = result.stdout.split()
    if len(parts) >= 3:
        return parts[2]
    raise RuntimeError(
        "Could not find the Windows host IP. Set CARLA_HOST "
        "(e.g. export CARLA_HOST=172.21.0.1) and start CarlaUE4.exe."
    )


def connect_client(
    host: str | None = None,
    port: int = 2000,
    timeout: float = 10.0,
) -> carla.Client:
    """Connect to CARLA and return a client with timeout set."""
    if host is None:
        host = get_windows_host_ip()
    print(f"Connecting to CARLA server at {host}:{port} ...")
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client
