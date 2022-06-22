# import libraries used in child processes and/or main
# Getting a frame every second for scanning with AI.
# This allows the program to keep up with the live feed's pace
import multiprocessing


# TODO: object detection function
def scan_image(frame):
    #imported libraries only for object detection:
    from imageai.Classification.Custom import CustomImageClassification
    import os
    # run object detection on frame (or saved image if frame not working):
    execution_path = os.getcwd()
    prediction = CustomImageClassification()
    prediction.setModelTypeAsResNet50()
    prediction.setModelPath("model_here!")
    prediction.setJsonPath("bird_model_class.json")
    prediction.loadModel(num_objects=4)
    predictions, probabilities = prediction.predictImage(frame, result_count=4)
    for eachPrediction, eachProbability in zip(predictions, probabilities): # output results for testing
        print(eachPrediction , " : " , eachProbability)

    # if object(s) found, keep output image and remove input image. else, remove both.
    if(probabilities[0] > 60):
        



if __name__ == '__main__':
    # import libraries used only in main
    import cv2  # include opencv library functions in python
    from vidgear.gears import CamGear

    def birdIDer():
        imagesFolder = "saved_images"
        x = 1
        c = 0

        # Create an object to hold reference to livestream
        #cap = cv2.VideoCapture(0)
        options = {"CAP_PROP_FPS": 1}
        cap = CamGear(source='https://www.youtube.com/watch?v=JhajwzEv1Fo',
                      stream_mode=True, logging=True, **options).start()
        # currently only doing one frame per 5 seconds.

        while True:
            frame = cap.read()  # capture the frame from live video
            # if (nowTime - startTime) >= fpsLimit: #without this, every single frame is saved even though options state 1fps
            if(c % 30 == 0):
                filename = imagesFolder + "/image_" + str(x) + ".jpg"
                cv2.imwrite(filename, frame)  # save image
                # new object detection process per frame
                p = multiprocessing.Process(target=scan_image(x))
                p.start()
                x += 1
            c += 1

    def main():
        birdIDer()

    main()
