from webcam.client import WebcamClient
from webcam.processors import flip_horizontal, draw_facemesh

with WebcamClient() as cam:
    cam.record(duration=10, processors=[flip_horizontal, draw_facemesh])
