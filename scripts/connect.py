"""
Minimal CARLA client connectivity check.

Run the CARLA server on Windows first (CarlaUE4.exe), then run this
script from WSL to confirm the client can reach it.
"""
from utils.connection import connect_client


def main() -> None:
    client = connect_client()
    world = client.get_world()

    print(f"Connected. Client version: {client.get_client_version()}")
    print(f"Server version: {client.get_server_version()}")
    print(f"Current map: {world.get_map().name}")

    blueprint_count = len(world.get_blueprint_library())
    print(f"Blueprint library has {blueprint_count} entries.")


if __name__ == "__main__":
    main()
