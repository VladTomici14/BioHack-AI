import keras.applications.mobilenet_v2
from PIL import Image
import numpy as np
import cv2
import os

from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model
from keras.utils import img_to_array

model = load_model("dataset/BIRDS-450-(200 X 200)-99.28.h5")
print(model.keys())

image = Image.open("dataset/train/APAPANE/002.jpg").resize((200, 200))
image = img_to_array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

pred = model.predict(image)[0]
pred_classes = pred.argmax(axis=-1)

print(f"pred_classes: {pred_classes}")

labels = []
for dir in os.listdir("dataset/train"):
    if os.path.isdir(f"dataset/train/{dir}"):
        labels.append(dir)

labels = sorted(labels)
