import random

import keras.applications.mobilenet_v2
from PIL import Image
import numpy as np
import random
import cv2
import os

from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model
from keras.utils import img_to_array

from classes import CLASSES_NAMES

model = load_model("dataset/BIRDS-450-(200 X 200)-99.28.h5")

image = Image.open("dataset/train/AFRICAN OYSTER CATCHER/001.jpg").resize((200, 200))
image = img_to_array(image)
image = image / 200.0
image = np.expand_dims(image, axis=0)

pred = model.predict(image)
pred = np.argmax(pred,axis=1)

# Map the label
labels = np.array(os.listdir("dataset/train"))
labels = [v for v in labels]
pred = [labels[k] for k in pred]

# Display the result
print(f'The first 5 predictions: {pred[:5]}')

# result_percentage = 0.0
# k = 0
# position = 0
# for prediction in pred:
#     prediction_percentage = float(prediction * 100000) / 1000
#     if prediction_percentage > result_percentage:
#         result_percentage = prediction_percentage
#         position = k
#     k += 1
#
# print(f"result_percentage: {result_percentage}")
# print(f"Result: {CLASSES_NAMES[position]}")

# for dir in os.listdir("dataset/train"):
#     if os.path.isdir(f"dataset/train/{dir}"):
#         labels.append(dir)
#
# labels = sorted(labels)
# print(f"label: {labels[pred_classes]}")



# species = random.sample(range(0, 450), 50)
# species = sorted([labels[x] for x in species])

# print(species)