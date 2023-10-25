import os
import numpy as np
from cv2 import *
from face_recognition import *
from pickle import *

#Run this program while running the program for the first time
#And also if your adding new images to the train data image folder
def known_faces(directory):
    known_faces_id=[]
    known_faces_encodings=[]
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jfif") and not filename.endswith(".py"):
                img=load_image_file(subdir + os.sep + filename)
                img_encode=face_encodings(img)
                filename1 = filename.split('.')[0]
                known_faces_id.append(filename1)
                known_faces_encodings.append(img_encode[0])
    return known_faces_id,known_faces_encodings
known_faces_id,known_faces_encodings=known_faces(r"C:\\Users\bhimi\\OneDrive\\Pictures\\Saved Pictures")
filename = 'data.pickle'
outfile = open(filename,'wb')
dump(known_faces_id,outfile)
dump(known_faces_encodings,outfile)
outfile.close()