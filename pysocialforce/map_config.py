from math import dist
from typing import List, Tuple
from dataclasses import dataclass, field

import numpy as np

Vec2D = Tuple[float, float]
Line2D = Tuple[float, float, float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]
Zone = Tuple[Vec2D, Vec2D, Vec2D] # rect ABC with sides |A B|, |B C| and diagonal |A C|


def sample_zone(zone: Zone, num_samples: int) -> List[Vec2D]:
    a, b, c = zone
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec_ba, vec_bc = a - b, c - b
    rel_width = np.random.uniform(0, 1, (num_samples, 1))
    rel_height = np.random.uniform(0, 1, (num_samples, 1))
    points = b + rel_width * vec_ba + rel_height * vec_bc
    return [(x, y) for x, y in points]


@dataclass
class Obstacle:
    vertices: List[Vec2D]
    lines: List[Line2D] = field(init=False)
    vertices_np: np.ndarray = field(init=False)

    def __post_init__(self):
        if not self.vertices:
            raise ValueError('No vertices specified for obstacle!')

        self.vertices_np = np.array(self.vertices)
        edges = list(zip(self.vertices[:-1], self.vertices[1:])) \
            + [(self.vertices[-1], self.vertices[0])]
        edges = list(filter(lambda l: l[0] != l[1], edges)) # remove fake lines that are just points
        lines = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = lines

        if not self.vertices:
            print('WARNING: obstacle is just a single point that cannot collide!')


@dataclass
class GlobalRoute:
    spawn_id: int
    goal_id: int
    waypoints: List[Vec2D]
    spawn_zone: Rect
    goal_zone: Rect

    def __post_init__(self):
        if self.spawn_id < 0:
            raise ValueError('Spawn id needs to be an integer >= 0!')
        if self.goal_id < 0:
            raise ValueError('Goal id needs to be an integer >= 0!')
        if len(self.waypoints) < 1:
            raise ValueError(f'Route {self.spawn_id} -> {self.goal_id} contains no waypoints!')

    @property
    def sections(self) -> List[Tuple[Vec2D, Vec2D]]:
        return [] if len(self.waypoints) < 2 else list(zip(self.waypoints[:-1], self.waypoints[1:]))

    @property
    def section_lengths(self) -> List[float]:
        return [dist(p1, p2) for p1, p2 in self.sections]

    @property
    def section_offsets(self) -> List[float]:
        lengths = self.section_lengths
        offsets = []
        temp_offset = 0.0
        for section_length in lengths:
            offsets.append(temp_offset)
            temp_offset += section_length
        return offsets

    @property
    def total_length(self) -> float:
        return 0 if len(self.waypoints) < 2 else sum(self.section_lengths)


@dataclass
class MapDefinition:
    obstacles: List[Obstacle]
    routes: List[GlobalRoute]
    crowded_zones: List[Zone]
