import numpy as np
import cv2

video_capture_0 = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()


    if (ret0):
        # Display the resulting frame
        cv2.imshow('Cam 0', frame0)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
cv2.destroyAllWindows()
