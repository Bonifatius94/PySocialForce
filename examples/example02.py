"""
# Example 02: test the configuration of the simulator
"""
import pysocialforce as pysf

obstacle01 = pysf.Obstacle(
    [(10, 10), (15,10), (15, 15), (10, 15)])
obstacle02 = pysf.Obstacle(
    [(20, 10), (25,10), (25, 15), (20, 15)])

route01 = pysf.GlobalRoute(
    [(0, 0), (10, 10), (20, 10), (30, 0)])
crowded_zone01 = ((10, 10), (20, 10), (20, 20))

map_def = pysf.MapDefinition(
    obstacles=[obstacle01, obstacle02],
    routes=[route01],
    crowded_zones=[crowded_zone01])

simulator = pysf.Simulator_v2(map_def)

for step in range(10):
    simulator.step()
    print(f"step {step}")
    print(simulator.states.ped_positions)
    print("=================")
