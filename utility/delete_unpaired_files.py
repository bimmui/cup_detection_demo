import os
import re
from shutil import copyfile
import argparse
import math
import random
from PIL import Image

path1 = r'C:\Users\Daniel\Documents\Tensorflow\workspace\training_demo\images\annotated_unsorted_photos'

files_to_deleted = (f for f in os.listdir(path1))

def checklist(list):
    x = None
    for picture in piclist:
        if os.path.exists(picture):
            x = picture
        else:
            pass
    return x

for filename in files_to_deleted:
    if filename.endswith(".xml"):
        root, _ = os.path.splitext(filename)
        xml_file = os.path.join(path1, filename)

        jpg2 = os.path.join(path1, f'{root}.jpeg')
        jpg3 = os.path.join(path1, f'{root}.JPEG')
        png = os.path.join(path1, f'{root}.png')
        png1 = os.path.join(path1, f'{root}.PNG')
        piclist = [jpg2, jpg3, png, png1]

        picture = checklist(piclist)
        if picture == None:
            print("deleted ", xml_file)
            os.remove(xml_file)
        else:
            if os.path.exists(picture) and os.path.exists(xml_file):
                pass
            else:
                if os.path.exists(picture) == True and os.path.exists(xml_file) == False:
                    print("deleted ", picture)
                    os.remove(picture)
    else:
        root, _ = os.path.splitext(filename)
        xml_file = os.path.join(path1, f'{root}.xml')
        picture = os.path.join(path1, filename)


        if os.path.exists(picture) and os.path.exists(xml_file):
            pass
        else:
            if os.path.exists(picture) == True and os.path.exists(xml_file) == False:
                print("deleted ", picture)
                os.remove(picture)
            elif os.path.exists(picture) == False and os.path.exists(xml_file) == True:
                print("deleted ", xml_file)
                os.remove(xml_file)
            else:
                pass
