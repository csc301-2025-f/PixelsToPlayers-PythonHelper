import os
import time
import csv
import cv2
from datetime import datetime


# -- configurations -- #
#TODO
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# -- Video save settings --#
CURR_DIR = os.path.dirname(__file__)
VIDEO_SAVE_PATH = os.path.join(CURR_DIR, "recordings")


def initialize_webcam():
    """Initializing webcam (video capture) object and returing it"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap

def record_video(cap, duration=10):
    """
    Recording video using <cap> of duration <duration> to save video in "recording file" (for now)
    """
    filename = os.path.join(
        VIDEO_SAVE_PATH,
        f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # video format
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = cv2.VideoWriter(filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #TODO: Loop runs with timer for now, will change to event later
    start_time = time.time()
    while time.time() - start_time < duration:
        
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the webcam on screen
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        # saving to video file
        out.write(frame)


    # release capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"Video saved to {filename}")
    return filename