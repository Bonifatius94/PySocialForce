"""
# Example 03: simulate with visual rendering
"""
import pysocialforce as pysf
import numpy as np

map_def = pysf.load_map("./maps/default_map.json")
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
    state = pysf.VisualizableSimState(step, ped_pos, actions)
    sim_view.render(state, fps=10)

sim_view.exit()
