# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from typing import List, Tuple
from warnings import warn

import numpy as np

from pysocialforce.utils import DefaultConfig, Config
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces


Line2D = Tuple[float, float, float, float]


class Simulator:
    def __init__(self, forces: List[forces.Force],
                 state: np.ndarray, groups: List[List[int]]=None,
                 obstacles: List[Line2D]=None, config: Config=DefaultConfig()):
        self.config = config
        # TODO: load obstacles from config
        self.scene_config = self.config.sub_config("scene")
        # initiate obstacles
        resolution: float = self.config("resolution", 10.0)
        self.env = EnvState(obstacles, resolution)

        # initiate agents
        self.peds = PedState(state, groups, self.config)

        # initiate forces
        self.forces = forces
        for force in self.forces:
            force.init(self, self.config)

    def compute_forces(self):
        """compute forces"""
        return sum(map(lambda x: x.get_force(), self.forces))

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
