from .client import WebcamClient
from .processors import draw_facemesh, flip_horizontal


with WebcamClient() as cam:
    cam.record(duration=10, processors=[flip_horizontal, draw_facemesh])
