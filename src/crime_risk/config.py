from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    grid_size: int = 30
    near_repeat_km: float = 1.5
    near_repeat_hours: int = 72
    decay_half_life_hours: float = 24.0
    spatial_sigma_km: float = 1.2
    lookback_hours: int = 14 * 24
    epsilon: float = 1e-6
