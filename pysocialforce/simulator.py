# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""

from __future__ import annotations
from typing import List, Tuple, Callable
from warnings import warn

import numpy as np

import pysocialforce as pysf
from pysocialforce.utils.config import SimulatorConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces


Line2D = Tuple[float, float, float, float]


def make_forces(sim: pysf.Simulator, config: pysf.utils.SimulatorConfig) -> List[pysf.forces.Force]:
    """Initialize forces required for simulation."""
    enable_group = config.scene_config.enable_group
    force_list = [
        pysf.forces.DesiredForce(config.desired_force_config, sim.peds),
        pysf.forces.SocialForce(config.social_force_config, sim.peds),
        pysf.forces.ObstacleForce(config.obstacle_force_config, sim),
    ]
    group_forces = [
        pysf.forces.GroupCoherenceForceAlt(config.group_coherence_force_config, sim.peds),
        pysf.forces.GroupRepulsiveForce(config.group_repulsive_force_config, sim.peds),
        pysf.forces.GroupGazeForceAlt(config.group_gaze_force_config, sim.peds),
    ]
    return force_list + group_forces if enable_group else force_list


class Simulator:
    def __init__(self, state: np.ndarray,
                 groups: List[List[int]]=None,
                 obstacles: List[Line2D]=None,
                 config: SimulatorConfig=SimulatorConfig(),
                 make_forces: Callable[[Simulator, SimulatorConfig], List[forces.Force]]=make_forces):
        self.config = config
        resolution = self.config.scene_config.resolution
        self.env = EnvState(obstacles, resolution)
        self.peds = PedState(state, groups, self.config.scene_config)
        self.forces = make_forces(self, config)

    def compute_forces(self):
        """compute forces"""
        return sum(map(lambda x: x(), self.forces))

    @property
    def current_state(self) -> Tuple[np.ndarray, List[List[int]]]:
        return self.peds.state, self.peds.groups

    def get_states(self):
        """Expose whole state"""
        warn('This function retrieves the whole state history \
              instead of just the most recent state',
              DeprecationWarning, stacklevel=2)
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    def get_raw_obstacles(self):
        return self.env.obstacles_raw

    def step_once(self):
        """step once"""
        self.peds.step(self.compute_forces())

    def step(self, n=1):
        """Step n time"""
        for _ in range(n):
            self.step_once()
        return self
