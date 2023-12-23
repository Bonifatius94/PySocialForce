# import os
# import numpy as np
# import pysocialforce as psf
# from pysocialforce.utils.plot import SceneVisualizer, SimRecording

# OUTPUT_DIR = "images/"

# if not os.path.exists(OUTPUT_DIR):
#     os.mkdir(OUTPUT_DIR)


# def test_crossing():
#     initial_state = np.array([[0.0, 0.0, 0.5, 0.5, 10.0, 10.0], [10.0, 0.3, -0.5, 0.5, 0.0, 10.0],])
#     rec = SimRecording()
#     s = psf.Simulator(initial_state, on_step=rec.append_frame)
#     s.step(50)
#     rec.static_obstacles = s.get_obstacles()

#     with SceneVisualizer(rec, OUTPUT_DIR + "crossing") as sv:
#         sv.animate()


# def test_narrow_crossing():
#     initial_state = np.array([[0.0, 0.0, 0.5, 0.5, 2.0, 10.0], [2.0, 0.3, -0.5, 0.5, 0.0, 10.0],])
#     rec = SimRecording()
#     s = psf.Simulator(initial_state, on_step=rec.append_frame)
#     s.step(40)
#     rec.static_obstacles = s.get_obstacles()

#     with SceneVisualizer(rec, OUTPUT_DIR + "narrow_crossing") as sv:
#         sv.animate()


# def test_opposing():
#     initial_state = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 10.0], [-0.3, 10.0, -1.0, 0.0, -0.3, 0.0],])
#     rec = SimRecording()
#     s = psf.Simulator(initial_state, on_step=rec.append_frame)
#     s.step(21)
#     rec.static_obstacles = s.get_obstacles()

#     with SceneVisualizer(rec, OUTPUT_DIR + "opposing") as sv:
#         sv.animate()


# def test_2opposing():
#     initial_state = np.array(
#         [
#             [0.0, 0.0, 0.5, 0.0, 0.0, 10.0],
#             [0.6, 10.0, -0.5, 0.0, 0.6, 0.0],
#             [2.0, 10.0, -0.5, 0.0, 2.0, 0.0],
#         ]
#     )

#     rec = SimRecording()
#     s = psf.Simulator(initial_state, on_step=rec.append_frame)
#     s.step(40)
#     rec.static_obstacles = s.get_obstacles()

#     with SceneVisualizer(rec, OUTPUT_DIR + "2opposing") as sv:
#         sv.animate()
