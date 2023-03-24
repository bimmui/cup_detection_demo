import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import zipfile
import cv2
import time


from collections import defaultdict
from threading import Thread, Barrier
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageFile

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from stereovision import triangulation as tri
from stereovision import calibrate

SCRIPT_DIR = os.path.dirname(__file__)
PATH_TO_SAVED_MODEL = os.path.join(SCRIPT_DIR, "exported-models/my_model2/saved_model")
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(SCRIPT_DIR, "annotations/labelmap.pbtxt"),
                                                                        use_display_name=True)
# Stereo vision setup parameters
## Using specs for two Logitech HD Webcam C270 at https://support.logi.com/hc/en-us/articles/360023462093-Logitech-HD-Webcam-C270-Technical-Specifications
## Horizonal FOV found using https://www.chiefdelphi.com/t/lifecam-hd-3000-specifications-horizontal-and-vertical-fov/353550 (saved using wayback machine)
FRAME_RATE = 30     #Camera frame rate (maximum at 30 fps)
BASE = 9            #Distance between the cameras [cm]
F = 4               #Camera lense's focal length [mm] 
ALPHA = 48.5        #Camera field of view in the horizontal plane [degrees]
print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn2 = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


class VideoStreamWidget():
    """ This is a VideoStreamWidget class. It modularizes
        the task of setting up a video stream, running the
        inference model on an image, and outputting the inference result
        as an np array on a pyplot graph. """

    def __init__(self, src, cam_name, model):
        self.cam_name = cam_name
        self.frame = None
        self.bbimg = None
        self.object_detected = False
        self.framenum = 0
        self.model = model
        self.capture = cv2.VideoCapture(src)
        self.obj_center_point = None
        self.imgsave_path = os.path.join(SCRIPT_DIR, f"images/svtest_imgs/{cam_name}")
        # Creates a folder for each VideoStreamWidget to save images in if folder is nonexistant
        if not os.path.exists(self.imgsave_path):
            os.makedirs(self.imgsave_path)
        
    def update(self):
        # Read the next frame from the stream in a different thread
        if self.capture.isOpened():
            ret, self.frame = self.capture.read()
            #self.frame = calibrate.undistortRectifyIndividually(self.frame)
            self.show_frame()
            time.sleep(.01)

    def frame_increase(self):
        self.framenum+=1

    # Uses model to run inference on the image
    def run_inference_for_single_image(self, image):
        # Configures the paths for saving the image and converts image to RGB (OpenCV is BGR)
        img_path = os.path.join(self.imgsave_path, f"img{self.framenum}.png")

        # Saves the current fram as an image, converts to RGB img, and then converts to numpy array
        cv2.imwrite(img_path, self.frame)
        time.sleep(.01)
        image = Image.open(img_path).convert(mode='RGB')
        image = np.asarray(image)
        self.frame_increase()

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
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

        
    # Makes use of the output dictionary after running inference on an image
    # to viualize results of the inference on a matplotlib graph
    # Returns tuple containing the image as a np_array and the output dict
    def show_inference(self):
        # Take the frame from webcam feed and convert that to np_array
        image_np = np.array(self.frame)

        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np)

        # Grabs the first detection from list and finds the coordinates of the bbox center point
        raw_bbox = output_dict['detection_boxes'][0]

        h, w, c = self.frame.shape
                        #min x                   #min y                  #max x              #max y
        boundBox = int(raw_bbox[0] * w), int(raw_bbox[1] * h), int(raw_bbox[2] * w), int(raw_bbox[3] * h)
        print("xmin:", boundBox[0], "ymin:", boundBox[1], "width:", boundBox[2], "height:", boundBox[3])
        self.obj_center_point = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        # Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            min_score_thresh=0.5,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=5)

        return image_np, output_dict

    def show_frame(self):
        # Calls show_inference to initiate running inference of single frame
        Imagenp, thedict = self.show_inference()
        
        # The image with a bounding box that can be displayed with cv2.imshow
        self.bbimg = Imagenp


if __name__ == '__main__':
    cam1 = VideoStreamWidget(2, "ret1", detect_fn) #left cam
    cam2 = VideoStreamWidget(0, "ret2", detect_fn2) #right cam
    
    while True:
        cam1.update()
        cam2.update()

        depth = tri.find_depth(cam2.obj_center_point, cam1.obj_center_point, cam2.frame, cam1.frame, BASE, F, ALPHA)

        cv2.putText(cam2.bbimg, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        cv2.putText(cam1.bbimg, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.

        cv2.imshow("frame right", cam2.bbimg) 
        cv2.imshow("frame left", cam1.bbimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break