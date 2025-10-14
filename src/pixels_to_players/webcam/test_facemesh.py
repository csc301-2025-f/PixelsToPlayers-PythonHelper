from webcam.client import WebcamClient
from webcam.processors import flip_horizontal, FaceMeshLogger

logger = FaceMeshLogger()

with WebcamClient() as cam:
    cam.record(duration=5, processors=[flip_horizontal, logger])

logger.save()
