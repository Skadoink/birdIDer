# Getting a frame every second for scanning with AI.
# This allows the program to keep up with the live feed's pace
# import libraries used in at least the child process:
import datetime
import tensorflow as tf
import time
import numpy as np
import multiprocessing
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import os
import imageio
from multiprocessing import Queue
from multiprocessing import Process
import pandas as pd
import csv

#return list of category names
def make_cats():
    cats = []
    cat_file = open("C:/Users/oskae/Documents/python/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt", "r")
    count = 2 #start at 2 because names are every 5th line from line 3. 
    for line in cat_file:
        count += 1
        if count % 5 != 0:
            continue
        name = line[11:-2]
        cats.append(name)
    return cats

# @tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    current_time = time.time()
    detections = detection_model(image, training=False)
    #print("Detections: " + str(detections))
    print("time for detections: " + str(time.time() - current_time))
    return detections, tf.reshape(shapes, [-1])


def scan_image(q):
    # Load our model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(
        "C:/Users/oskae/Documents/python/TensorFlow/workspace/training_demo/exported-models/my_model/pipeline.config")
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(
        "C:/Users/oskae/Documents/python/TensorFlow/workspace/training_demo/exported-models/my_model/checkpoint/", 'ckpt-0')).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap("C:/Users/oskae/Documents/python/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt",
                                                                        use_display_name=True)

    #make a list of the categories
    cats = make_cats()

    # run object detection
    starttime = time.time()
    count = 0

    while True:
        while q.empty():
            time.sleep(0.01)  # prevent overly frequent checking

        currenttime = time.time()
        count += 1
        image_np = q.get()
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, shapes = detect_fn(input_tensor, detection_model)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes']
                [0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        #print info about each detection above score threshold 
        i = 0
        while detections.get("detection_scores")[0][i] >= 0.3:
            image_name = "image" + str(count)
            category_name = cats[int(detections["detection_classes"][0][i])]
            probability = (detections["detection_scores"][0][i]).numpy()
            print(image_name + ": species: " + category_name + " probability: " + str(probability))
            with open("records.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now(), category_name, probability, image_name])
    
            i += 1

        # save image with detections
        # this can eventually use filters to only save images with detections
        imagesFolder = "programs/man_IDer/detected_images"
        filename = imagesFolder + "/image_" + str(count) + ".jpg"
        #cv2.imwrite(filename, image_np_with_detections)
        imageio.imwrite(filename, image_np_with_detections)  # save image
        print("time taken image processing: " +
                str(time.time() - currenttime))
        # check we are saving one image every second:
        print("attempt save image " + str(count) + " at " +
                str(time.time() - starttime))



if __name__ == '__main__':
    # import libraries used only in main
    from vidgear.gears import CamGear

    def stream_parser():
        imagesFolder = "saved_images"
        x = 0
        c = 0
        starttime = time.time()

        # Create an object to hold reference to livestream
        options = {"CAP_PROP_FPS": 1}
        cap = CamGear(source='https://www.youtube.com/watch?v=N609loYkFJo',
                      stream_mode=True, logging=True, **options).start()
        q = Queue()
        p = Process(target=scan_image, args=(q,))
        p.start()

        while True:
            image_np = cap.read()  # capture the frame from live video
            if(c % 30 == 0):  # every 30th frame
                q.put(image_np)
                x += 1

            c += 1

    def main():
        stream_parser()

    main()
