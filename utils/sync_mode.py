"""Context manager for CARLA synchronous simulation + Traffic Manager."""
from __future__ import annotations

from types import TracebackType

import carla


class SynchronousMode:
    def __init__(
        self,
        client: carla.Client,
        world: carla.World,
        fps: float = 20.0,
        tm_port: int = 8000,
        no_rendering: bool = False,
    ) -> None:
        self.client = client
        self.world = world
        self.fps = fps
        self.tm_port = tm_port
        self.no_rendering = no_rendering
        self.traffic_manager: carla.TrafficManager | None = None
        self._settings: carla.WorldSettings | None = None

    def __enter__(self) -> tuple[carla.World, carla.TrafficManager]:
        self._settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        settings.no_rendering_mode = self.no_rendering
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        return self.world, self.traffic_manager

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)
        if self._settings is not None:
            self.world.apply_settings(self._settings)
