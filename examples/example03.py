"""
# Example 03: simulate with visual rendering
"""
import pysocialforce as pysf
import numpy as np

obstacle01 = pysf.map_config.Obstacle(
    [(10, 10), (15,10), (15, 15), (10, 15)])
obstacle02 = pysf.map_config.Obstacle(
    [(20, 10), (25,10), (25, 15), (20, 15)])

route01 = pysf.map_config.GlobalRoute(
    [(0, 0), (10, 10), (20, 10), (30, 0)])
crowded_zone01 = ((10, 10), (20, 10), (20, 20))

map_def = pysf.map_config.MapDefinition(
    obstacles=[obstacle01, obstacle02],
    routes=[route01],
    crowded_zones=[crowded_zone01])

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
