import cv2
import numpy as np
import os
import sqlite3

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("recognizer/trainingdata.yml")

def getprofile(id):
    conn = sqlite3.connect("sqlite.db")
    cmd = "SELECT * FROM STUDENTS WHERE ID = ?"
    cursor = conn.execute(cmd, (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        profile = getprofile(id)
        print(profile)
        if (profile != None):
            cv2.putText(img, "Name: " + str(profile[0]), (x-w//2 + 40,y + h + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,127), 2)
            cv2.putText(img, "Msv: " + str(profile[1]), (x-w//2 + 40,y + h + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,127), 2)
            cv2.putText(img, "Lop: " + str(profile[2]), (x-w//2 + 40, y + h + 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Age: " + str(profile[3]), (x-w//2 + 40, y + h + 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
    cv2.imshow("FACE", img)
    if (cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()