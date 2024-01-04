"""
# Example 02: test the configuration of the simulator
"""
import pysocialforce as psf


simulator = psf.Simulator_v2()

for step in range(10):
    simulator.step()
    print(simulator)