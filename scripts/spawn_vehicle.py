"""
Spawn a vehicle at a map spawn point, then clean it up on exit.

Requires the CARLA server (CarlaUE4.exe) running on Windows.
"""
import time

from utils.connection import connect_client
from utils.spectator import follow_vehicle
from utils.vehicles import spawn_vehicle


def main() -> None:
    client = connect_client()
    world = client.get_world()
    vehicle = None

    print(f"Map: {world.get_map().name}")
    print(f"Client {client.get_client_version()} / Server {client.get_server_version()}")

    try:
        vehicle = spawn_vehicle(world)
        world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick()
        print(f"Spawned {vehicle.type_id} (id={vehicle.id})")
        print(f"Location: {vehicle.get_transform().location}")

        vehicle.set_autopilot(True)
        print("Autopilot enabled. Spectator following for 20s (Ctrl+C to stop early)...")
        end = time.time() + 20.0
        while time.time() < end:
            world.wait_for_tick()
            follow_vehicle(world, vehicle)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if vehicle is not None and vehicle.is_alive:
            print(f"Destroying vehicle id={vehicle.id}")
            vehicle.destroy()


if __name__ == "__main__":
    main()
