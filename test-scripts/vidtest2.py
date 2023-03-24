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
from threading import Thread, Barrier
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageFile

# sys.path.append("..")

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

## debugging imports and functions
# import datetime
# from sys import getsizeof
# def get_thread_position(thread):
#     frame = sys._current_frames().get(thread.ident, None)
#     if frame:
#         return frame.f_code.co_filename, frame.f_code.co_name, frame.f_code.co_firstlineno, thread.name


SCRIPT_DIR = os.path.dirname(__file__)
PATH_TO_SAVED_MODEL = os.path.join(SCRIPT_DIR, "exported-models/my_model2/saved_model")
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(SCRIPT_DIR, "annotations/labelmap.pbtxt"),
                                                                        use_display_name=True)
print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn2 = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
#print('The models takes up', int(getsizeof(detect_fn)+getsizeof(detect_fn2)), 'bytes of space')

# Used to prevent overloading the model with constant input from individual VideoStreamWidgets

class VideoStreamWidget():
    """ This is a VideoStreamWidget class. It modularizes
        the task of setting up a video stream, running the
        inference model on an image, and outputting the inference result
        as an np array on a pyplot graph. """

    def __init__(self, src, cam_name, model):
        self.cam_name = cam_name
        self.framenum = 0
        self.model = model
        self.capture = cv2.VideoCapture(src)
        ## for calculating fps
        # self._start = None
        # self._end = None
        self.imgsave_path = os.path.join(SCRIPT_DIR, f"images/vidtest_imgs/{cam_name}")
        # Creates a folder for each VideoStreamWidget to save images in if folder is nonexistant
        if not os.path.exists(self.imgsave_path):
            os.makedirs(self.imgsave_path)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (ret, self.frame) = self.capture.read()
                self.show_frame()
                time.sleep(.01)
        ## Use this code when calculating fps for each cam
        # self._start = datetime.datetime.now()
        # while self.framenum < 300:
        #     if self.capture.isOpened():
        #         (ret, self.frame) = self.capture.read()
        #     self.show_frame()
        #     time.sleep(.01)
        # self._end = datetime.datetime.now()
        # print(f"This is {self.cam_name}", self.fps())

    def frame_increase(self):
        self.framenum+=1
        print(self.framenum)

    # def start(self):
	# 	# start the timer
    #     #self._start = datetime.datetime.now()
    #     #return self

    # def stop(self):
	# 	# stop the timer
    #     #self._end = datetime.datetime.now()

    # def elapsed(self):
	# 	# return the total number of seconds between the start and
	# 	# end interval
    #     return (self._end - self._start).total_seconds()

    def fps(self):
		# compute the (approximate) frames per second
        return self.framenum / self.elapsed()

    # Uses model to run inference on the image
    def run_inference_for_single_image(self, image):
        # Configures the paths for saving the image and converts image to RGB (OpenCV is BGR)
        img_path = os.path.join(self.imgsave_path, f"img{self.framenum}.png")

        # Saves the current fram as an image, converts to an RGB img, and then converts to numpy array
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

        # Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=5)

        # Below is used fot checking detection score before saving image
        #if output_dict['detection_scores'][0] < 0.2:
        #  print("This is the score: ", output_dict['detection_scores'][0])
        # save = True
        output_dict['category_index'] = category_index
        return(image_np, output_dict)

    def show_frame(self):
        # Calls show_inference to initiate running inference of single frame
        Imagenp, thedict = self.show_inference()
        cv2.imshow(self.cam_name, cv2.resize(Imagenp, (800,600)))
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)


## debugging and performance tests using cProfile
# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     main()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()

## the "main" main
if __name__ == '__main__':
    cam1 = VideoStreamWidget(0, "ret1", detect_fn)
    cam2 = VideoStreamWidget(1, "ret2", detect_fn2)


## the old "main", keep if needed to be scraped for future purposes
#Now we open the webcam and start detecting objects
# video_capture = cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)
# count = 0
# while True:
#     # Capture frame-by-frame
#     re,frame = video_capture.read()
#     img_path = os.path.join(SCRIPT_DIR, f"ioms{count}.png")
#     cv2.imwrite(img_path, frame)
#     print("image saved")
#     Imagenp, thedict = show_inference(detect_fn, frame)
#     cv2.imshow('object detection', cv2.resize(Imagenp, (800,600)))
#     if keyboard.is_pressed('space'):
#         print(thedict)
#if tosave == True:
#        pass
#    else:
#        os.remove(f"ioms{count}.png")
# #        print("removed")
#     count+=1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#video_capture.release()
#cv2.destroyAllWindows()
