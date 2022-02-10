import cv2
import numpy as np
import tensorflow as tf
from app.main import logger
from app.utils import get_classnames_dict, stringToRGB, cv_to_base64
from PIL import Image

DETECTION_THRESHOLD = 0.3
model_path = "models/evoa-face-object-detector/evoa-face-object-detector.tflite"
classes = get_classnames_dict()

# Load the labels into a list
# classes = ['???'] * model.model_spec.config.num_classes
# label_map = model.model_spec.config.label_map
# for label_id, label_name in label_map.as_dict().items():
#   classes[label_id-1] = label_name
  
# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


def preprocess_image(image, input_size):
    img = stringToRGB(image)
    logger.debug(img.shape)
    original_image = img
    # resized_img = tf.image.resize(img, input_size)
    # resized_img = resized_img[tf.newaxis, :]
    # resized_img = tf.cast(resized_img, dtype=tf.uint8)
    resized_img = cv2.resize(img, input_size)
    logger.debug(resized_img.shape)
    return resized_img, original_image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    # print(interpreter.get_output_details())
    output_details = interpreter.get_output_details()[index]
    logger.debug(output_details)
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()
  logger.debug(signature_fn)
  # # logger.debug("detect_objects - signature_fn: %s" % signature_fn)
  # # # Feed the input image to the model
  # # output = signature_fn(images=image)
  # logger.debug(interpreter.get_signature_list())
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  logger.debug(input_details)

  input_shape = input_details[0]['shape']
  # logger.debug(input_shape)
  input_tensor = np.array(np.expand_dims(image, 0), dtype=np.uint8)
  logger.debug(input_tensor.shape)
  # input_index = interpreter.get_input_details()[0]['index']
  # logger.debug(input_index)
  # interpreter.set_tensor(input_index, input_tensor)
  # interpreter.invoke()
  
  # logger.debug("Get output")
  # output_details = interpreter.get_input_details()
  # logger.debug(output_details)
  # # Get all outputs from the model
  # scores = get_output_tensor(interpreter, 0)
  # logger.debug(scores)
  # boxes = get_output_tensor(interpreter, 1)
  # logger.debug(boxes)
  # count = int(get_output_tensor(interpreter, 2))
  # logger.debug(count)
  # classes = get_output_tensor(interpreter, 3)
  # logger.debug(classes)

  # logger.debug(threshold)
  
  # Feed the input image to the model
  output = signature_fn(images=input_tensor)
  logger.debug(output)
  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  logger.debug(results)
  return results

def run_odt_and_draw_results(image, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  logger.debug(interpreter.get_input_details())
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  logger.debug("height: %s, width: %s" % (input_height, input_width))
  logger.debug(interpreter.get_signature_list())
  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image,
      (input_height, input_width)
    )
  logger.debug(preprocessed_image.shape)
  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
  logger.debug(results)
  # Plot the detection results on the input image
  # original_image_np = original_image.numpy().astype(np.uint8)
  original_image_np = original_image
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  logger.debug(original_uint8.shape)
  # Convert to Base64
  original_b64 = cv_to_base64(original_uint8)
  return original_b64

def load_interpreter():
    """ Load the TFLite model """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logger.debug("Loaded model: %s" % interpreter.get_input_details())
    return interpreter

def predict(image, interpreter):
    logger.debug(interpreter.get_signature_list())
    detection_result_image = run_odt_and_draw_results(
    image,
    interpreter,
    threshold=DETECTION_THRESHOLD)
    
    return detection_result_image
