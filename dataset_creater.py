import cv2
import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insertorupdate(id, name, msv, lop, age):
    conn = sqlite3.connect("sqlite.db")
    cmd = "SELECT * FROM STUDENTS WHERE ID = " + str(id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if (isRecordExist == 1):
        conn.execute("UPDATE STUDENTS SET Name = ? WHERE ID = ?", (name, id))
        conn.execute("UPDATE STUDENTS SET Msv = ? WHERE ID = ?", (msv, id))
        conn.execute("UPDATE STUDENTS SET Lop = ? WHERE ID = ?", (lop, id))
        conn.execute("UPDATE STUDENTS SET Age = ? WHERE ID = ?", (age, id))
    else:
        conn.execute("INSERT INTO STUDENTS (Name, Msv, Lop, Age, Id) VALUES(?,?,?,?,?)", (name, msv, lop, age, id))

    conn.commit()
    conn.close()

#insert user defined values into table

id = input('Enter User Id: ')
name = input('Enter user name: ')
msv = input('Enter user msv: ')
lop = input('Enter user lop: ')
age = input('Enter user age: ')

insertorupdate(id, name, msv, lop, age)

#detect face in web camera coding

sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataset/user." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum >= 20:
        break
cam.release()
cv2.destroyAllWindows()