from webcam import initialize_webcam, record_video

def main():

    # Starting webcam and recording
    cap = initialize_webcam()
    record_video(cap, 5)

main()
