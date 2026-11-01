"""
Interactive TUI to pick and load a CARLA map.

Requires the CARLA server (CarlaUE4.exe) running on Windows.

  python -m scripts.change_map
"""
from __future__ import annotations

import sys
import termios
import tty

from utils.connection import connect_client

# Map loads can take a while on large towns.
LOAD_TIMEOUT = 120.0

CLEAR = "\033[H\033[J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def short_name(map_path: str) -> str:
    return map_path.rsplit("/", 1)[-1]


def current_map_name(client) -> str:
    return short_name(client.get_world().get_map().name)


def read_key() -> str:
    """Read one keypress. Returns 'up', 'down', 'enter', 'quit', or a char."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch in ("\x03", "\x04", "q", "Q"):
            return "quit"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "k":
            return "up"
        if ch == "j":
            return "down"
        if ch != "\x1b":
            return ch
        seq = sys.stdin.read(2)
        if seq in ("[A", "OA"):
            return "up"
        if seq in ("[B", "OB"):
            return "down"
        return "quit"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def render(maps: list[str], cursor: int, current: str, status: str) -> None:
    lines = [
        "CARLA Map Picker",
        f"Current: {current}",
        status or "",
        "",
    ]
    for idx, name in enumerate(maps):
        pointer = ">" if idx == cursor else " "
        marker = "*" if name == current else " "
        lines.append(f" {pointer}{marker} {name}")
    lines.extend(["", "↑↓ move   Enter load   q quit"])
    sys.stdout.write(CLEAR + "\n".join(lines) + "\n")
    sys.stdout.flush()


def run_picker(client, maps: list[str], current: str) -> None:
    cursor = maps.index(current) if current in maps else 0
    status = ""
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()
    try:
        while True:
            render(maps, cursor, current, status)
            key = read_key()
            if key == "quit":
                break
            if key == "up":
                cursor = (cursor - 1) % len(maps)
                status = ""
            elif key == "down":
                cursor = (cursor + 1) % len(maps)
                status = ""
            elif key == "enter":
                chosen = maps[cursor]
                if chosen == current:
                    status = f"{chosen} is already loaded."
                    continue
                status = f"Loading {chosen} ... this can take a minute."
                render(maps, cursor, current, status)
                try:
                    client.set_timeout(LOAD_TIMEOUT)
                    client.load_world(chosen)
                    current = current_map_name(client)
                    status = f"Loaded {current}."
                except RuntimeError as exc:
                    status = f"Failed: {exc}"
    finally:
        sys.stdout.write(SHOW_CURSOR + "\n")
        sys.stdout.flush()


def prompt_numbered(maps: list[str], current: str) -> str | None:
    print(f"Current: {current}\n")
    for i, name in enumerate(maps, start=1):
        marker = "*" if name == current else " "
        print(f" {i:2d}) {marker} {name}")
    print("\nEnter a number, or q to quit.")
    choice = input("> ").strip()
    if choice.lower() in {"", "q", "quit"}:
        return None
    try:
        idx = int(choice)
    except ValueError:
        print("Not a number.")
        return None
    if not 1 <= idx <= len(maps):
        print("Out of range.")
        return None
    return maps[idx - 1]


def main() -> None:
    client = connect_client(timeout=LOAD_TIMEOUT)
    print("Fetching available maps...")
    maps = sorted(short_name(m) for m in client.get_available_maps())
    current = current_map_name(client)
    if not maps:
        print("Server reported no maps.")
        return

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        chosen = prompt_numbered(maps, current)
        if chosen is None or chosen == current:
            return
        print(f"Loading {chosen} ...")
        client.load_world(chosen)
        print(f"Loaded {current_map_name(client)}.")
        return

    try:
        run_picker(client, maps, current)
    except KeyboardInterrupt:
        print("\nCancelled.")


if __name__ == "__main__":
    main()
