from collections import namedtuple
from typing import Any, List, Optional

# Holds scene metadata (e.g., name, dt), but without the memory
# footprint of all the actual underlying scene data.
SceneMetadata = namedtuple("SceneMetadata", ["env_name", "name", "dt", "raw_data_idx"])


class Scene:
    """Holds the data for a particular scene."""

    def __init__(
        self,
        dt: float,
        name: str,
        location: str,
        length_timesteps: int,
        raw_data_idx: int,
        data_access_info: Any,
        description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.location = location
        self.dt = dt
        self.length_timesteps = length_timesteps
        self.raw_data_idx = raw_data_idx
        self.data_access_info = data_access_info
        self.description = description

    def length_seconds(self) -> float:
        return self.length_timesteps * self.dt

    def __repr__(self) -> str:
        return "/".join([self.env_name, self.name])
