# import libraries
from vidgear.gears import CamGear 
# Getting a frame every second for scanning with AI.
# This allows the program to keep up with the live feed's pace

import cv2  # include opencv library functions in python
import time  # for sleep


# TODO: object detection function
def scan_image(frame):
    # run AI on frame
    # if object found, save frame and log result.
    return

def birdIDer():
    fpsLimit = 1  # throttle limit
    startTime = time.time()
    imagesFolder = "saved_images"
    x = 1

    # Create an object to hold reference to livestream
    #cap = cv2.VideoCapture(0)
    options = {"CAP_PROP_FPS":1}
    cap = CamGear(source='https://www.youtube.com/watch?v=JhajwzEv1Fo', stream_mode = True, logging=True, **options).start()

    while True:
        nowTime = time.time()
        frame = cap.read()  # capture a frame from live video
        if (int(nowTime - startTime)) > fpsLimit:
            # check whether frame is successfully captured
            # TODO: unique naming might be needed
            filename = imagesFolder + "/image_" + str(x) + ".jpg"; x += 1
            # save image, may need to be removed later.
            cv2.imwrite(filename, frame)
            scan_image(frame)

            startTime = time.time()  # reset time

def main():
    birdIDer()

main()