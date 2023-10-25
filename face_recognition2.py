import numpy as np
from cv2 import *
from face_recognition import *
import pickle

#Run this program where your face is under proper light
def recognition():
    with open('data.pickle','rb') as f:
        known_faces_id = pickle.load(f)
        known_faces_encodings = pickle.load(f)
    video_capture=cv2.VideoCapture(0)
    i=1
    #empty list for the appending names 
    names_list=[]
    while True:
        #Reading the each frame
        ret,small_frame=video_capture.read()
        rgb_small_frame = small_frame[:, :, ::-1]
        #locate the face locations in the frame
        face_location=face_locations(rgb_small_frame)
        test_face_encodings = face_encodings(rgb_small_frame ,face_location)
        for (top,right,bottom,left),face_encoding in zip(face_location,test_face_encodings):
            matches=compare_faces(known_faces_encodings,face_encoding,tolerance=0.5)
            name="Unkown Person"
            #Matching the face with the known face
            face_distances=face_distance(known_faces_encodings,face_encoding)
            best_match_index=np.argmin(face_distances)
            if matches[best_match_index]:
                name=known_faces_id[best_match_index]
                names_list.append(name)
                
            rectangle(small_frame,(left,top),(right,bottom),(0,255,0),2)
            rectangle(small_frame,(left,bottom-20),(right,bottom),(0,0,0),cv2.FILLED)
            putText(small_frame,name,(left+1,bottom-1),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),1)
        cv2.imshow('face_recognizing',small_frame)
        if waitKey(1)==27:#order of esc key
            break
        #i=i+1
    video_capture.release()
    cv2.destroyAllWindows()
    #If you don't wanna accuracy you can just return the name
    #return name
    return names_list

#This function is for accuracy for like between two similar look like people or if any improper conditions where camera can't look at exactly
def most_frequent_accuracy():
    names_list= recognition()
    counter = 0
    name = names_list[0] 
      
    for i in names_list: 
        curr_frequency = names_list.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            name = i 
  
    return name
print(most_frequent_accuracy())