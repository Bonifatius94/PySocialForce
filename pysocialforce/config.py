"""Config"""
from dataclasses import dataclass
from pysocialforce.ped_population import PedSpawnConfig


@dataclass
class SceneConfig:
    enable_group: bool = True
    agent_radius: float = 0.35
    dt_secs: float = 0.1
    max_speed_multiplier: float = 1.3
    tau: float = 0.5
    resolution: float = 10


@dataclass
class GroupCoherenceForceConfig:
    factor: float = 3.0


@dataclass
class GroupReplusiveForceConfig:
    factor: float = 1.0
    threshold: float = 0.55


@dataclass
class GroupGazeForceConfig:
    factor: float = 4.0
    fov_phi: float = 90.0


@dataclass
class DesiredForceConfig:
    factor: float = 1.0
    relaxation_time: float = 0.5
    goal_threshold: float = 0.2


@dataclass
class SocialForceConfig:
    factor: float = 5.1
    lambda_importance: float = 2.0
    gamma: float = 0.35
    n: int = 2
    n_prime: int = 3
    activation_threshold: float = 20.0


@dataclass
class ObstacleForceConfig:
    factor: float = 10.0
    sigma: float = 0.0
    threshold: float = -0.57


@dataclass
class SimulatorConfig:
    scene_config: SceneConfig = SceneConfig()
    group_coherence_force_config: GroupCoherenceForceConfig = GroupCoherenceForceConfig()
    group_repulsive_force_config: GroupReplusiveForceConfig = GroupReplusiveForceConfig()
    group_gaze_force_config: GroupGazeForceConfig = GroupGazeForceConfig()
    desired_force_config: DesiredForceConfig = DesiredForceConfig()
    social_force_config: SocialForceConfig = SocialForceConfig()
    obstacle_force_config: ObstacleForceConfig = ObstacleForceConfig()
    ped_spawn_config: PedSpawnConfig = PedSpawnConfig()
