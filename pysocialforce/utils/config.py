"""Config"""
from dataclasses import dataclass, fields
import toml


@dataclass
class SceneConfig:
    enable_group: bool = True
    agent_radius: float = 0.35
    step_width: float = 1.0
    max_speed_multiplier: float = 1.3
    tau: float = 0.5
    resolution: float = 10


@dataclass
class GoalAttractiveForceConfig:
    factor: float=1.0


@dataclass
class PedRepulsiveForceConfig:
    factor: float = 1.5
    v0: float = 2.1
    sigma: float = 0.3
    fov_phi: float = 100.0
    fov_factor: float = 0.5 # out of view factor


@dataclass
class SpaceRepulsiveForceConfig:
    factor: float = 1
    u0: float = 10
    r: float = 0.2


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


@dataclass
class ObstacleForceConfig:
    factor: float = 10.0
    sigma: float = 0.0
    threshold: float = -0.57


class DCUnpack:
    """Helper class to initialize dataclass instances from a dictionary.
    See https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary"""
    class_field_cache = {}

    @classmethod
    def new(cls, class_to_instantiate, arg_dict):
        if class_to_instantiate not in cls.class_field_cache:
            cls.class_field_cache[class_to_instantiate] = \
                {f.name for f in fields(class_to_instantiate) if f.init}

        field_set = cls.class_field_cache[class_to_instantiate]
        filtered_arg_dict = {k : v for k, v in arg_dict.items() if k in field_set}
        return class_to_instantiate(**filtered_arg_dict)


@dataclass
class SimulatorConfig:
    scene_config: SceneConfig = SceneConfig()
    goal_attractive_force_config: GoalAttractiveForceConfig = GoalAttractiveForceConfig()
    ped_repulsive_force_config: PedRepulsiveForceConfig = PedRepulsiveForceConfig()
    space_repulsive_force_config: SpaceRepulsiveForceConfig = SpaceRepulsiveForceConfig()
    group_coherence_force_config: GroupCoherenceForceConfig = GroupCoherenceForceConfig()
    group_repulsive_force_config: GroupReplusiveForceConfig = GroupReplusiveForceConfig()
    group_gaze_force_config: GroupGazeForceConfig = GroupGazeForceConfig()
    desired_force_config: DesiredForceConfig = DesiredForceConfig()
    social_force_config: SocialForceConfig = SocialForceConfig()
    obstacle_force_config: ObstacleForceConfig = ObstacleForceConfig()

    def load_from_toml_file(self, config_file: str):
        data = toml.load(config_file)
        if 'scene' in data:
            self.scene_config = DCUnpack.new(SceneConfig, data['scene'])
        if 'goal_attractive_force' in data:
            self.goal_attractive_force_config = DCUnpack.new(GoalAttractiveForceConfig, data['goal_attractive_force'])
        if 'ped_repulsive_force' in data:
            self.ped_repulsive_force_config = DCUnpack.new(PedRepulsiveForceConfig, data['ped_repulsive_force'])
        if 'space_repulsive_force' in data:
            self.space_repulsive_force_config = DCUnpack.new(SpaceRepulsiveForceConfig, data['space_repulsive_force'])
        if 'group_coherence_force' in data:
            self.group_coherence_force_config = DCUnpack.new(GroupCoherenceForceConfig, data['group_coherence_force'])
        if 'group_repulsive_force' in data:
            self.group_repulsive_force_config = DCUnpack.new(GroupReplusiveForceConfig, data['group_repulsive_force'])
        if 'group_gaze_force' in data:
            self.group_gaze_force_config = DCUnpack.new(GroupGazeForceConfig, data['group_gaze_force'])
        if 'desired_force' in data:
            self.desired_force_config = DCUnpack.new(DesiredForceConfig, data['desired_force'])
        if 'social_force' in data:
            self.social_force_config = DCUnpack.new(SocialForceConfig, data['social_force'])
        if 'obstacle_force' in data:
            self.obstacle_force_config = DCUnpack.new(ObstacleForceConfig, data['obstacle_force'])
