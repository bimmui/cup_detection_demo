import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2, time
import keyboard


from collections import defaultdict
from threading import Thread
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# sys.path.append("..")

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# PATH_TO_SAVED_MODEL = "C:/Users/Daniel/Documents/Tensorflow/workspace/training_demo/exported-models/my_model2" + "/saved_model"
SCRIPT_DIR = os.path.dirname(__file__)
PATH_TO_SAVED_MODEL = os.path.join(SCRIPT_DIR, "exported-models/my_model2/saved_model")
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(SCRIPT_DIR, "annotations/labelmap.pbtxt"),
                                                                    use_display_name=True)
# print('Loading model...', end='')
# start_time = time.time()
#
# # Load saved model and build the detection function
# detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
#
# end_time = time.time()
# elapsed_time = end_time - start_time
# print('Done! Took {} seconds'.format(elapsed_time))

class VideoStreamWidget():
    def __init__(self, src, cam_name):
        self.cam_name = cam_name
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.imgsave_path = os.path.join(SCRIPT_DIR, f"images/vidtest_imgs/{cam_name}")
        # Creates a folder for each VideoStreamWidget to save images in if folder is nonexistant
        if not os.path.exists(self.imgsave_path):
            os.makedirs(self.imgsave_path)

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.cam_name, self.frame) = self.capture.read()
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow(self.cam_name, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

def run_inference_for_single_image(model, image):
  bigpath = "C:/Users/Daniel/Documents/Tensorflow/workspace/training_demo/"
  endpath = f"ioms{count}.png"
  finalpath = bigpath + endpath
  image = Image.open(finalpath).convert(mode='RGB')
  image = np.asarray(image)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict

def show_inference(model, frame):
  save = False
  #take the frame from webcam feed and convert that to array
  image_np = np.array(frame)
  # Actual detection.

  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=5)

  #if output_dict['detection_scores'][0] < 0.2:
    #  print("This is the score: ", output_dict['detection_scores'][0])
     # save = True
  output_dict['category_index'] = category_index
  return(image_np, output_dict)

#Now we open the webcam and start detecting objects
# video_capture = cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)
# count = 0
if __name__ == '__main__':
    cam1 = VideoStreamWidget(0, "ret1")
    cam2 = VideoStreamWidget(1, "ret2")
    while True:
        try:
            # Capture frame-by-frame
            cam1.show_frame()
            cam2.show_frame()
        except AttributeError:
            pass
while True:
    # Capture frame-by-frame
    re,frame = video_capture.read()
    img_path = os.path.join(SCRIPT_DIR, f"ioms{count}.png")
    cv2.imwrite(img_path, frame)
    print("image saved")
    Imagenp, thedict = show_inference(detect_fn, frame)
    cv2.imshow('object detection', cv2.resize(Imagenp, (800,600)))
    if keyboard.is_pressed('space'):
        print(thedict)
#if tosave == True:
#        pass
#    else:
#        os.remove(f"ioms{count}.png")
#        print("removed")
    count+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
