
from pickle import FALSE
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
import csv
# from PIL import ImageGrab

path = 'ImagesAttendance/'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
df=pd.DataFrame(columns=["Names","Time"])
print(len(images))
d1=""

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        encodeList.append(encode)
    return encodeList

timeList=[]
nameList=[]
x=0
def markAttendance(name):
    today=datetime.now()
    d1=today.strftime("%d/%m/%Y %H:%M:%S")
    
    n = len(nameList)
    f = True
    for i in range(n):
        if name==nameList[i]:
            f = False
            break
    if f:
        nameList.append(name)
        timeList.append(d1)
    

#### FOR CAPTURING SCREEN (DEMO)
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr


encodeListKnown=findEncodings(images)
print('Encoding Complete',len(encodeListKnown))

cap = cv2.VideoCapture(0)
print(encodeListKnown)
while True:
    success, img = cap.read()

# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    i=0
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        print(encodesCurFrame)
        faceDis = []
        matches = []
        for i in range (len(encodeListKnown)):
            
            xy = face_recognition.face_distance(encodeListKnown[i], encodeFace)
            yz = face_recognition.compare_faces(encodeListKnown[i], encodeFace)
            faceDis.append(xy)
            matches.append(yz)
            i+=1
        #matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        
        print(matches)
        print(len(faceDis)) #error in faceDis
        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    frame = {'Names': nameList, 'Time': timeList}
    # df["Names"]=pd.Series(nameList)
    # df["Time"]=pd.Series(timeList)
    df = pd.DataFrame(frame);
    df.to_csv("Attendance.csv")
    print("1")
    print(df)
    print("2")
    print(nameList)
    print(timeList)
    nameSz = len(nameList)
    #for i in range(nameSz):


    # series2=pd.Series(timeList)
    # series1=pd.Series(nameList)

    # df.Name=series1
    # df.Time=series2
    
    
    cv2.imshow('Webcam', img)
    # data=[]
    # data_in=[]
    # nt = len(nameList)
    # for i in range(nt):
    #     data_in = [nameList[i],timeList[i]]
    #     data.append(data_in)
    # with open('Attendance.csv','w',newline='') as fp:
    #     a = csv.writer(fp, delimeter=',')
    # a.writerows(data)
    # print(data)

    cv2.waitKey(1)