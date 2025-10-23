from pixels_to_players.webcam.client import WebcamClient
from pixels_to_players.webcam.processors import draw_facemesh, flip_horizontal

import unittest

class WebcamTests(unittest.TestCase):
    def test_something(self):
        with WebcamClient() as cam:
            path = cam.record(duration=10, processors=[flip_horizontal, draw_facemesh])
        self.assertEqual(True, path.exists())


if __name__ == '__main__':
    unittest.main()



