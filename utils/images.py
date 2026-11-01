"""Image conversion helpers for CARLA camera sensors."""
import carla
import numpy as np


def image_to_array(image: carla.Image) -> np.ndarray:
    """Convert a CARLA RGB camera image to an HxWx3 uint8 RGB numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA -> BGR
    return array[:, :, ::-1]  # BGR -> RGB
