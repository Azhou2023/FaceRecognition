import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import random
import cv2
import uuid
from mtcnn import MTCNN
from PIL import Image

siamese_model = keras.models.load_model("siameseModel")

def preprocess(file_path):
    
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    
    img = tf.image.resize(img, (100,100))
    img = img / 255.0

    return img

detector = MTCNN()
def detectFace(filePath, size=(100,100)):
    pixels = plt.imread(filePath)
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]
    
    return face

#validate that a picture of my face if it passes a similarity threshold for a certain percentage of test images
def verify(model, verification_threshold, detection_threshold):
    
    results = []
    for image in os.listdir(os.path.join('Testing', 'Verification')):
        input_img = preprocess(os.path.join('Testing', 'Input', 'input_img.jpg'))
        validation_img = preprocess(os.path.join('Testing', 'Verification', image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    
    verification = detection / len(os.listdir(os.path.join('Testing', 'Verification'))) 
    verified = verification >= verification_threshold
    
    return results, verified, verification
        
#live verification with opencv
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame = frame[120:120+250, 200:200+250, :]
    cv2.imshow("frame", frame)  
    
    if cv2.waitKey(1) == ord("a"):
        cv2.imwrite(os.path.join("Testing", "UnprocessedInput", "input_img.jpg"), frame)
        faceArray = detectFace(os.path.join("Testing", "UnprocessedInput", "input_img.jpg"))
        img = Image.fromarray(faceArray)
        img.save(os.path.join("Testing", "Input", "input_img.jpg"))
        results, verified, verification = verify(siamese_model,  0.5, 0.5)
        print(verified) 
        print("results:", results)
        print(verification)
        
    if cv2.waitKey(1) == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
