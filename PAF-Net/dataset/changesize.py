import cv2 as cv
import os


path='dataset/picture/data/image/'
path2='dataset/picture/data/label/'
for i in range(100):
    imgpth=path+str(i)+'.jpg'
    if os.path.exists(imgpth):
        img=cv.imread(imgpth)
        img=cv.resize(img,(256,256))
        gth=cv.imread(path2+str(i)+'.png')
        gth=cv.resize(gth,(256,256))
        cv.imwrite(imgpth,img)
        cv.imwrite(path2+str(i)+'.png',gth)
