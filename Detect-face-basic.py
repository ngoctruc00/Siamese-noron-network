
# Import the necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


def detect_faces(cascade, test_image, scaleFactor=1.2):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    # convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        image_copy = image_copy[y:y + h, x:x + w]
    return image_copy


path = 'data/pretrain/s3'
path_save = 'data/train/s3'
files = os.listdir(path)
for index, file in enumerate(files):
    test_image2 = cv2.imread(os.path.join(path, file))
    faces = detect_faces(haar_cascade_face, test_image2)
    # plt.imshow(convertToRGB(faces))
    cv2.imwrite(os.path.join(path_save, file), faces)
    print('Done image:'+str(index))





