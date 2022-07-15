# Getting a frame every second for scanning with AI.
# This allows the program to keep up with the live feed's pace
# import libraries used in at least the child process:
import tensorflow as tf
import time

# @tf.function


def detect_fn(image, detection_model):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    current_time = time.time()
    detections = detection_model(image, training=False)
    print("time for detections: " + str(time.time() - current_time))

    return detections, tf.reshape(shapes, [-1])


# TODO adjust scan_image to accept queue for input, run constantly (while true), imports, and maybe more.
def scan_image(q):
    # run object detection on frame (or saved image if frame not working):
    starttime = time.time()
    count = 0

    while True:
        if q.empty():
            time.sleep(0.01)  # prevent unnecessarily frequent checking
        else:
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

            # Display output
            #cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            # save image with detections
            imagesFolder = "programs/man_IDer/detected_images"
            filename = imagesFolder + "/image_" + str(count) + ".jpg"
            #cv2.imwrite(filename, image_np_with_detections)
            imageio.imwrite(filename, image_np_with_detections) # save image
            print("time taken for whole function: " +
                  str(time.time() - starttime))

            # cv2.destroyAllWindows()


if __name__ == '__main__':
    from object_detection.builders import model_builder
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.utils import config_util
    from object_detection.utils import label_map_util
    import os
    import tensorflow as tf
    import numpy as np
    import multiprocessing
    import time
    import imageio
    from multiprocessing import Queue
    from multiprocessing import Process

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

    # import libraries used only in main
    from vidgear.gears import CamGear

    def birdIDer():
        imagesFolder = "saved_images"
        x = 1
        c = 0
        starttime = time.time()

        # Create an object to hold reference to livestream
        options = {"CAP_PROP_FPS": 1}
        cap = CamGear(source='https://www.youtube.com/watch?v=JhajwzEv1Fo',
                      stream_mode=True, logging=True, **options).start()
        q = Queue()
        p = Process(scan_image(q))
        p.start()

        while True:
            image_np = cap.read()  # capture the frame from live video
            if(c % 30 == 0):  # every 30th frame
                q.put(image_np)
                x += 1  # DELETEME
                # check we are saving one image every second:
                print("saved image " + str(x) + " at " +
                      str(time.time() - starttime))
            c += 1

    def main():
        birdIDer()

    main()
