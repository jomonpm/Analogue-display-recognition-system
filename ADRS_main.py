import cv2
import requests
import time
import urllib.request
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
import glob
import math
import time
import matplotlib.pyplot as plt
import threading

# Function to load an image into a numpy array
def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    return np.array(image)

# Load category index from a label map file
category_index = label_map_util.create_category_index_from_labelmap("D:\\Learn\THD\\Intelligent Systems\\Labelled Data\\Data_v02\\Needle-Zero-Referencepoint_label_map.pbtxt", use_display_name=True)

# Load the model from a saved model directory
ex_path = 'D:\Learn\THD\Intelligent Systems\Model_R01\inference_graph2\saved_model'
model = tf.saved_model.load(ex_path)

# Function to run inference for a single image
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

# Main function for image processing and measurements
def process_images_and_measurements():
    while True:
        try:
            stream_url1 = urllib.request.urlopen('http://192.168.7.174/capture')
            img1 = np.array(bytearray(stream_url1.read()), dtype=np.uint8)
            img = cv2.imdecode(img1, -1)
            cv2.imshow('title', img)

            # Object Detection
            image_np = cv2.resize(img, (640, 640))
            output_dict = run_inference_for_single_image(model, image_np)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=min_score_thresh
            )

            # ... (filtering and measurement calculation code)

            print(measurement)

            # Function to transfer the value of measurement to GUI
            def update_value():
                global measurement

            value_thread = threading.Thread(target=update_value)
            value_thread.start()

            # Saves the image after detection
            timestamp = int(time.time())
            filename = save_folder + f"image_{timestamp}.jpg"
            cv2.imwrite(filename, image_np)

            if cv2.waitKey(1) == ord('q'):
                break

            time.sleep(capture_interval)

        except Exception as e:
            print(f'Error: {e}')
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define global variables here (e.g., min_score_thresh, capture_interval, save_folder)
    min_score_thresh = 0.9
    capture_interval = 2
    save_folder = "D:/Photos/images/"

    # Start the main image processing and measurement function
    process_images_and_measurements()
