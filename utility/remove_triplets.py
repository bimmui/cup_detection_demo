#there existed a .png and .jpg of a file when al i needed was the .png
#this script removes the .jpg if the .png version of it existed


import os
import re
from shutil import copyfile
import argparse
import math
import random
from PIL import Image




path1 = r'C:\Users\Daniel\Documents\Tensorflow\workspace\training_demo\images\annotated_unsorted_photos'

files_to_deleted = (f for f in os.listdir(path1))
for filename in files_to_deleted:
	root, _ = os.path.splitext(filename)
	if filename.endswith(".jpg") or filename.endswith(".JPG"):
		if os.path.exists(os.path.join(path1, f'{root}.png')) or os.path.exists(os.path.join(path1, f'{root}.PNG')):
			os.remove(os.path.join(path1, filename))
