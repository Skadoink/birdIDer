# import libraries used in child processes and/or main
# Getting a frame every second for scanning with AI.
# This allows the program to keep up with the live feed's pace
import multiprocessing

def scan_image(image_np, x):
    #imported libraries only for object detection:
    import numpy as np
    import tensorflow as tf
    import os

    # Load our model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import config_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder

    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file("C:/Users/oskae/Documents/python/TensorFlow/workspace/training_demo/exported-models/my_model/pipeline.config")
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join("C:/Users/oskae/Documents/python/TensorFlow/workspace/training_demo/exported-models/my_model/checkpoint/", 'ckpt-0')).expect_partial()

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    category_index = label_map_util.create_category_index_from_labelmap("C:/Users/oskae/Documents/python/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt",
                                                                    use_display_name=True)

    # run object detection on frame (or saved image if frame not working):
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    # Display output
    #cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    #save image with detections
    imagesFolder = "detected_images"
    filename = imagesFolder + "/image_" + str(x) + ".jpg"
    cv2.imwrite(filename, image_np_with_detections)  # save image

    #cv2.destroyAllWindows()
    # if object(s) found, keep output image and remove input image. else, remove both:
    



if __name__ == '__main__':
    # import libraries used only in main
    import numpy.core.multiarray
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

        while True:
            image_np = cap.read()  # capture the frame from live video
            if(c % 30 == 0): #every 30th frame 
                # filename = imagesFolder + "/image_" + str(x) + ".jpg"
                # cv2.imwrite(filename, frame)  # save image
                # new object detection process per frame
                p = multiprocessing.Process(target=scan_image(image_np, x))
                p.start()
                x += 1
            c += 1

    def main():
        birdIDer()

    main()