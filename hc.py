import numpy as np
import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mo = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

def findd(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ipd=0
    faces = face_cascade.detectMultiScale(gray, 1.3, 15)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 9)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 7)
        ipd=(eyes[0][0]+eyes[0][2]//2)-(eyes[1][0]+eyes[1][2]//2)
        m = mo.detectMultiScale(roi_gray, 1.5, 15)
        area_l=[]
        for (ex, ey, ew, eh) in m:
            area_l.append(eh*ew)
        x=min(area_l)
        pos=area_l.index(x)
        x,y,w,h=m[pos][0],m[pos][1],m[pos][2],m[pos][3]
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 0, 255), 7)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img,int(ipd)