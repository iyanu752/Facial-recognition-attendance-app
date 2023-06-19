import cv2
import numpy as np
import face_recognition
import os

path = 'ImagesAttendance'
images = []
classNames = []

myList = os.listdir(path)
print(myList)
for cl in myList:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
         facedis = face_recognition.face_distance(encodeListKnown, encodeFace)
         print(facedis)
         matchIndex = np.argmin(facedis)

         if matches[matchIndex]:
             name = classNames[matchIndex].upper()
             print(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)




#facLoc = face_recognition.face_locations(imgElon)[0]
#encodeElon = face_recognition.face_encodings(imgElon)[0]
#cv2.rectangle(imgElon, (facLoc[3], facLoc[0]), (facLoc[1], facLoc[2]), (255,0,255),2)
#
#facLocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest, (facLocTest[3], facLocTest[0]), (facLocTest[1], facLocTest[2]), (255,0,255),2)
#
#results = face_recognition.compare_faces([encodeElon], encodeTest)
#faceDis = face_recognition.face_distance([encodeElon], encodeTest)