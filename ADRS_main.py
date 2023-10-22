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

# Load label map for object detection
def load_label_map(label_map_path):
    return label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Load the TensorFlow object detection model
def load_object_detection_model(model_path):
    return tf.saved_model.load(model_path)

# Perform object detection on a single image
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

# Function to filter detections based on a minimum score
def filter_detections(output_dict, min_score_thresh):
    filtered_detections = []

    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > min_score_thresh:
            detection = {
                'class': output_dict['detection_classes'][i],
                'score': output_dict['detection_scores'][i],
                'bbox': {
                    'xmin': output_dict['detection_boxes'][i][1] * 640,
                    'ymin': output_dict['detection_boxes'][i][0] * 640,
                    'xmax': output_dict['detection_boxes'][i][3] * 640,
                    'ymax': output_dict['detection_boxes'][i][2] * 640
                }
            }
            filtered_detections.append(detection)

    return filtered_detections

# Measurement Calculation
def calculate_measurement(output_dict):
    zero_ymax = []
    zero_xmin = []

    for m in output_dict['detections']:
        if m["class"] == 3:
            zero_ymax.append(m['bbox']['ymax'])
            zero_xmin.append(m['bbox']['xmin'])

    zero_dis_ymax = max(zero_ymax)
    index = zero_ymax.index(zero_dis_ymax)
    y_max_del = []
    y_max_del.append(zero_dis_ymax)
    del zero_ymax[index]
    zero_dis_ymax = max(zero_ymax)
    y_max_del.append(zero_dis_ymax)

    xmin_list = []
    zero_dis_bbox = {}

    for m in output_dict['detections']:
        if m["class"] == 3:
            if m['bbox']['ymax'] == y_max_del[0] or m['bbox']['ymax'] == y_max_del[1]:
                xmin_list.append(m['bbox']['xmin'])

    zero_dis_x = min(xmin_list)

    for m in output_dict['detections']:
        if m["class"] == 3:
            if m['bbox']['xmin'] == zero_dis_x:
                zero_dis_y = m['bbox']['ymax']
                zero_dis_bbox = m['bbox']

    needle_dis_bbox = {}
    reference_dis_bbox = {}

    for m in output_dict['detections']:
        if m["class"] == 1:
            needle_dis_bbox = m['bbox']
        elif m["class"] == 2:
            reference_dis_bbox = m['bbox']

    needle_centre_x, needle_centre_y = (needle_dis_bbox['xmin'] + needle_dis_bbox['xmax']) / 2, (
        needle_dis_bbox['ymax'] + needle_dis_bbox['ymin']) / 2
    reference_centre_x, reference_centre_y = (reference_dis_bbox['xmin'] + reference_dis_bbox['xmax']) / 2, (
        reference_dis_bbox['ymax'] + reference_dis_bbox['ymin']) / 2
    zero_centre_x, zero_centre_y = (zero_dis_bbox['xmin'] + zero_dis_bbox['xmax']) / 2, (
        zero_dis_bbox['ymax'] + zero_dis_bbox['ymax']) / 2

    adj_side = zero_centre_y - reference_centre_y
    hyp_side = math.sqrt((zero_centre_x - reference_centre_x) ** 2 + (zero_centre_y - reference_centre_y) ** 2)
    angle = math.acos(adj_side / hyp_side)
    outer_angle = 2 * math.degrees(angle)
    inner_angle = 360 - outer_angle

    max_value_read = 120
    least_count = max_value_read / inner_angle

    vec_1 = [zero_centre_x - reference_centre_x, zero_centre_y - reference_centre_y]
    vec_2 = [needle_centre_x - reference_centre_x, needle_centre_y - reference_centre_y]

    dot_pro = (vec_1[0] * vec_2[0]) + (vec_1[1] * vec_2[1])
    mag_vec_1 = math.sqrt((zero_centre_x - reference_centre_x) ** 2 + (zero_centre_y - reference_centre_y) ** 2)
    mag_vec_2 = math.sqrt((needle_centre_x - reference_centre_x) ** 2 + (needle_centre_y - reference_centre_y) ** 2)

    angle_swept = math.acos(dot_pro / (mag_vec_1 * mag_vec_2))
    angle_swept_deg = math.degrees(angle_swept)
    measurement = angle_swept_deg * least_count

    return measurement

# Main function to perform measurements and display the image
def main():
    category_index = load_label_map("D:\\Learn\\THD\\Intelligent Systems\\Labelled Data\\Data_v02\\Needle-Zero-Referencepoint_label_map.pbtxt")
    model = load_object_detection_model('D:\\Learn\\THD\\Intelligent Systems\\Model_R01\\inference_graph2\\saved_model')

    image_folder = "D:/Photos/images/"
    min_score_thresh = 0.9
    measurements = []
    times = []
    filtered_detections = []
    capture_interval = 2
    save_folder = "D:/Photos/images/"

    while True:
        try:
            stream_url1 = urllib

