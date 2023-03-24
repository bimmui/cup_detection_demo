import cv2
import os

SCRIPT_DIR = os.path.dirname(__file__)

cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(0)

num = 0


while True:

    succes1, img = cap.read()
    succes2, img2 = cap2.read()
    
    
    rightcamdir = os.path.join(SCRIPT_DIR, 'images/stereoright/')
    leftcamdir = os.path.join(SCRIPT_DIR, 'images/stereoleft/')

    if not os.path.exists(rightcamdir):
        os.makedirs(rightcamdir)
    if not os.path.exists(leftcamdir):
        os.makedirs(leftcamdir)

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == 8: # wait for backspace key to save picture
        cv2.imwrite(os.path.join(leftcamdir, f'imageL{num}.png'), img)
        cv2.imwrite(os.path.join(rightcamdir, f'imageR{num}.png'), img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)
