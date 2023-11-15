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

# fill anchor and positive data in UnprocessedData with face pictures
def capture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imshow('Image', frame)
        
        if cv2.waitKey(1) == ord('a'):
            imgname = os.path.join("UnprocessedData/Positive", '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
            imgname = os.path.join("UnprocessedData/Anchor", '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)        
                
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# capture()

tf.config.run_functions_eagerly(True)

detector = MTCNN()
binaryCrossentropy = tf.losses.BinaryCrossentropy() 
optimizer = tf.keras.optimizers.Adam(0.0001)

#mtcnn to draw box around only the face and eliminate interference brom background
def detectFace(filePath, size=(100,100)):
    pixels = plt.imread(filePath)
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]
    
    return face


def processFaces(directory):
    for filename in os.listdir(os.path.join("UnprocessedData", directory)):
        faceArray = detectFace(os.path.join("UnprocessedData", directory, filename))
        img = Image.fromarray(faceArray)
        img.save(os.path.join("Data", directory, '{}.jpg'.format(uuid.uuid1())))
        os.remove(os.path.join("UnprocessedData", directory, filename))
        print("success")

processFaces("Anchor")
processFaces("Negative")
processFaces("Positive")


anchor = tf.data.Dataset.list_files('Data/Anchor'+'\*.jpg').take(3000)
anchor = anchor.shuffle(buffer_size=1582)

positive = tf.data.Dataset.list_files('Data/Positive'+'\*.jpg').take(3000)
positive = positive.shuffle(buffer_size=1570)

negative = tf.data.Dataset.list_files('Data/Negative'+'\*.jpg').take(3000)

    
def preprocess(file_path):
    
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    
    img = tf.image.resize(img, (100,100))
    img = img / 255.0

    return img


def preprocessTwin(inputImg, validationImg, label):
    return (preprocess(inputImg), preprocess(validationImg), label)

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

data = data.map(lambda img, img2, label: tf.py_function(preprocessTwin, inp=[img, img2, label], Tout=[tf.float32]))
data = data.shuffle(10000)

train_data = data.take(round(0.9*len(data)))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


def embedding():
    input = Input(shape=(100,100,3))
    
    c1 = Conv2D(64, (10,10), activation="relu")(input)
    p1 = MaxPooling2D(64, (2,2), padding="same")(c1)
    
    c2 = Conv2D(128, (7, 7), activation="relu")(p1)
    p2 = MaxPooling2D(64, (2,2), padding="same")(c2)
    
    c3 = Conv2D(128, (4,4), activation="relu")(p2)
    p3 = MaxPooling2D(64, (2,2), padding="same")(c3)
    
    c4 = Conv2D(256, (4,4), activation="relu")(p3)
    
    flatten = Flatten()(c4)
    output = Dense(4096, activation="sigmoid")(flatten)
    
    return Model(inputs=[input], outputs=[output])  

class Distance(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, input, validation):
        return tf.math.abs(input - validation)
    
embedding = embedding()

def siameseNetwork():
    
    inpImage = Input(name='input_img', shape=(100,100,3))
    valImage = Input(name='validation_img', shape=(100,100,3))
    
    inputEmbedding = embedding(inpImage)
    valEmbedding = embedding(valImage)
    
    distance = Distance(inputEmbedding, valEmbedding)
    
    output = Dense(1, activation="sigmoid")(distance)
    
    return Model(inputs=[inpImage, valImage], outputs=output)

siamese_network = siameseNetwork()

checkpointPrefix = os.path.join('./checkpoints', 'ckpt')
checkpoint = tf.train.Checkpoint(optimzer=optimizer, siamese_network=siamese_network)


def train(data, epochs):
    for epoch in range(1, epochs+1):
        print('\n Epoch {}/{}'.format(epoch, epochs))    
        
        for batch in data:
            with tf.GradientTape() as tape:     
                X = batch[:2]
                y = batch[2]
                
                yhat = siamese_network(X, training=True)
                loss = binaryCrossentropy(y, yhat)
        
            grad = tape.gradient(loss, siamese_network.trainable_variables)
            optimizer.apply_gradients(zip(grad, siamese_network.trainable_variables))

train(train_data, 50)
siamese_network.save("siameseModel")


        

    
        
      
        
        
