import cv2
import hc
import glob

ipd=[]

def abs(x):
    if x<0:
       x*=-1
    return x

for pic in glob.glob("face_img/*.jpg"):
    img=cv2.imread(pic)
    _,i=hc.findd(img)
    print(abs(i))
    ipd.append(abs(i))

print(ipd)