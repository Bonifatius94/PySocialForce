# coding=utf-8

"""Synthetic pedestrian behavior with social groups
simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""

from __future__ import annotations
from typing import List, Tuple, Callable
from warnings import warn

import numpy as np

import pysocialforce as pysf
from pysocialforce.map_config import MapDefinition
from pysocialforce.ped_behavior import PedestrianBehavior
from pysocialforce.ped_grouping import PedestrianStates, PedestrianGroupings
from pysocialforce.config import SimulatorConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces
from pysocialforce.ped_population import populate_simulation


Line2D = Tuple[float, float, float, float]
SimState = Tuple[np.ndarray, List[List[int]]]
EMPTY_MAP = MapDefinition([], [], [])
SimPopulator = Callable[[SimulatorConfig, MapDefinition],
                        Tuple[PedestrianStates, PedestrianGroupings, List[PedestrianBehavior]]]


def make_forces(sim: pysf.Simulator, config: SimulatorConfig) -> List[pysf.forces.Force]:
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


class Simulator_v2:
    def __init__(self,
                 map_definition: MapDefinition=EMPTY_MAP,
                 config: SimulatorConfig=SimulatorConfig(),
                 make_forces: Callable[[Simulator, SimulatorConfig], List[forces.Force]]=make_forces,
                 populate: SimPopulator=lambda s, m: \
                    populate_simulation(
                        s.scene_config.tau, s.ped_spawn_config,
                        m.routes, m.crowded_zones),
                 on_step: Callable[[SimState], None] = lambda s: None):
        self.config = config
        self.on_step = on_step
        self.states, self.groupings, self.behaviors = populate(config, map_definition)
        obstacles = [line for o in map_definition.obstacles for line in o.lines] \
            if map_definition.obstacles else None
        self.env = EnvState(obstacles, self.config.scene_config.resolution)
        self.peds = PedState(
            self.states.raw_states,
            self.groupings.groups,
            self.config.scene_config)
        self.forces = make_forces(self, config)

    @property
    def current_state(self) -> SimState:
        return self.peds.state, self.peds.groups

    @property
    def obstacles(self):
        return self.env.obstacles

    @property
    def raw_obstacles(self):
        return self.env.obstacles_raw

    def get_obstacles(self):
        return self.env.obstacles

    def get_raw_obstacles(self):
        return self.env.obstacles_raw

    def _step_once(self):
        """step once"""
        forces = sum(map(lambda force: force(), self.forces))
        self.peds.step(forces)
        for behavior in self.behaviors:
            behavior.step()

    def step(self, n=1):
        """Step n time"""
        for _ in range(n):
            self._step_once()
            self.on_step(self.current_state)
        return self


class Simulator:
    def __init__(self, state: np.ndarray,
                 groups: List[List[int]]=None,
                 obstacles: List[Line2D]=None,
                 config: SimulatorConfig=SimulatorConfig(),
                 make_forces: Callable[[Simulator, SimulatorConfig], List[forces.Force]]=make_forces,
                 on_step: Callable[[SimState], None] = lambda s: None):
        self.config = config
        self.on_step = on_step
        resolution = self.config.scene_config.resolution
        self.env = EnvState(obstacles, resolution)
        self.peds = PedState(state, groups, self.config.scene_config)
        self.forces = make_forces(self, config)

    def compute_forces(self):
        """compute forces"""
        return sum(map(lambda force: force(), self.forces))

    @property
    def current_state(self) -> SimState:
        return self.peds.state, self.peds.groups

    def get_states(self):
        warn('For performance reasons This function does not retrieve the whole \
              state history (it used to facilitate video recordings). \
              Please use the on_step callback for recording purposes instead!',
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
            self.on_step(self.current_state)
        return self
