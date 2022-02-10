import io
import os
from logging import log
import cv2
import base64
import numpy as np
import yaml
import requests

from io import BytesIO
from PIL import Image
from loguru import logger

import app.settings as ste

DENY_LIST = {}
ALLOW_LIST = {}

def get_rule_from_file(rule_list, filename):
    with open(filename) as f:
        docs = yaml.load_all(f, Loader=yaml.FullLoader)
        for doc in docs:
            for k,v in doc.items():
                rule_list[k] = v

def check_rule(rule_list, object_list):
    for object in object_list:
        for k,v in rule_list.items():
            if object in v:
                return True
    return False      
    
def stringToRGB(base64_string):
    """
    Take in base64 string and return cv image
    """
    img_stream = BytesIO(base64.b64decode(base64_string))
    return cv2.imdecode(np.fromstring(img_stream.read(), np.uint8),
                        cv2.IMREAD_COLOR)
    
def cv_to_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    logger.debug(f"buffer: {buffer}")
    jpg_as_text = base64.b64encode(buffer)
    b64 = jpg_as_text.decode("utf-8")
    return b64



def detector(npimage):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # # Loads the image into memory
    # with io.open(image, 'rb') as image_file:
    #     content = image_file.read()

    image = types.Image(content=npimage)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    _labels = []

    # print('Labels:')
    for label in labels:
        # print(label.description)
        _labels.append(label.description.lower())
        
    return _labels

def predict(image):
    pil_image = Image.fromarray(image)
    buff = BytesIO()
    pil_image.save(buff, format="JPEG")
    img_value = buff.getvalue()
    
    detected_object = detector(img_value)
    logger.debug(detected_object)
    deny_photo = check_rule(DENY_LIST, detected_object)
    allow_photo = check_rule(ALLOW_LIST, detected_object)
    logger.debug(f"deny: {deny_photo}, allow: {allow_photo}")
    
    verify_status = False
    if allow_photo and not deny_photo:
        verify_status = True
    
    output = {
        'verify_status': verify_status,
        'detected': detected_object,
    }
    return output


# def make_request(image, server_url):
#     """Send request to the Tensorflow Serving API

#     Args:
#         image ([type]): [description]
#         server_url ([type]): [description]
#     """
#     img_array = stringToRGB(image)
#     np_image = np.expand_dims(img_array, 0).tolist()
#     request_data = '{"instances" : %s}' % np_image
    
#     r = requests.post(server_url, data=request_data)
    
def get_classnames_dict():
    """ Get the classes instances from EVOA Face dataset """
    classes = {}
    i = 0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    classes_file = open(dir_path + "/evoa-face-labels.txt")
    for line in classes_file:
        classes[i] = line.split("\n")[0]
        i += 1
        
    return classes

# def get_prediction(image, server_url):
#     """ Get the filtered Prediction key from the TensorFlow request """
#     predictions = make_request(image, server_url)["predictions"][0]
#     classes_names = get_classnames_dict()
#     num_detections = int(predictions["num_detections"])
    
#     # Filtering out the unused predictions
#     detection_boxes = predictions["detection_boxes"][:num_detections]
#     detection_classes = predictions["detection_classes"][:num_detections]
#     detection_classes_names = []
#     for detection in detection_classes:
#         detection_classes_names.append(classes_names[detection - 1])
#     detection_scores = predictions["detection_scores"][:num_detections]
    
#     return {"num_detections": num_detections,
#             "detection_boxes": detection_boxes,
#             "detection_classes": detection_classes_names,
#             "detection_scores": detection_scores}