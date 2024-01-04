"""
# Example 01: basic import test
"""
import pysocialforce as psf


simulator = psf.Simulator_v2()

for step in range(10):
    simulator.step()
    print(simulator)
