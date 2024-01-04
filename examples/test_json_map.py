import json
import pysocialforce as pysf
from pysocialforce.map_config import Obstacle, GlobalRoute
import numpy as np


# Load the JSON file
with open('pysocialforce/maps/default_map.json', 'r') as file:
    map_json = json.load(file)

# Create the obstacles
obstacles = []
for obstacle in map_json['obstacles']:
    obstacles.append(pysf.map_config.Obstacle(obstacle))

# Create the routes
routes = []
for route in map_json['ped_routes']:
    routes.append(pysf.map_config.GlobalRoute(route['waypoints']))

# Create the crowded zones
crowded_zones = []
for crowded_zone in map_json['crowded_zone']:
    crowded_zones.append(tuple(crowded_zone))

# TODO field of view must be adapted to the map dimensions or the map dimensions must be added as a transparent obstacle sourrounding the map.

# ================== #

map_def = pysf.map_config.MapDefinition(
    obstacles=obstacles,
    routes=routes,
    crowded_zones=crowded_zones)

simulator = pysf.Simulator_v2(map_def)
sim_view = pysf.SimulationView(obstacles=map_def.obstacles, scaling=10)
sim_view.show()

for step in range(10_000):
    simulator.step()
    ped_pos = np.array(simulator.states.ped_positions)
    ped_vel = np.array(simulator.states.ped_velocities)
    actions = np.concatenate((
            np.expand_dims(ped_pos, axis=1),
            np.expand_dims(ped_pos + ped_vel, axis=1)
        ), axis=1)
    state = pysf.sim_view.VisualizableSimState(step, ped_pos, actions)
    sim_view.render(state, fps=10)

sim_view.exit()