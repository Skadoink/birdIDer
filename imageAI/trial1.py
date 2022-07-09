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

    # Create an object to hold reference to camera video capturing
    cap = cv2.VideoCapture(0)

    # check if connection with camera is successful
    if cap.isOpened():
        while True:
            nowTime = time.time()
            if (int(nowTime - startTime)) > fpsLimit:
                ret, frame = cap.read()  # capture a frame from live video
                frameId = cap.get(1)  # current frame number
                # check whether frame is successfully captured
                if ret:
                    # TODO: unique naming might be needed
                    filename = imagesFolder + "/image_" + \
                        str(int(frameId)) + ".jpg"
                    # save image, may need to be removed later.
                    cv2.imwrite(filename, frame)
                    scan_image(frame)

                # print error if frame capturing was unsuccessful
                else:
                    print("Error : Failed to capture frame")

                startTime = time.time()  # reset time

    # print error if the connection with camera is unsuccessful
    else:
        print("Cannot open camera")

def main():
    birdIDer