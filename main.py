import numpy as np
import keras
import cv2
import os

import tensorflow as tf
from keras_preprocessing.image import img_to_array

from utils.utils import Utils

# ------- reading the classes from the training directory -------
classes = []
for x in os.listdir("reptiles/dataset"):
    if os.path.isdir(f"reptiles/dataset/{x}"):
        classes.append(x)


def apply_model_to_img(img_path):
    image = cv2.imread(img_path)
    image = Utils().center_image(image)
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    cv2.imshow("snoui", image)
    cv2.waitKey(0)

    array_image = image.astype("float") / 224.0
    array_image = img_to_array(array_image)
    array_image = np.expand_dims(array_image, axis=0)

    model = keras.models.load_model("model_reptiles.h5")
    predictions = model.predict(array_image)[0]

    print(f"\n< ------ RESULTS: {img_path} ------ >")
    k = 0
    for prediction in predictions:
        prediction_percentage = float(prediction * 100000) / 1000
        print(f"{classes[k]}: {prediction_percentage}% \n")
        k += 1

    result = classes[np.argmax(predictions)]

    print(f"RESULT: {result}")


def draw_rect(image, box):
    y_min = int(max(1, (box[0] * image.height)))
    x_min = int(max(1, (box[1] * image.width)))
    y_max = int(min(image.height, (box[2] * image.height)))
    x_max = int(min(image.width, (box[3] * image.width)))

    # draw a rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

def apply_tflite_model_to_img(img_path, model_path):
    interpreter = tf.lite.Interpreter(model_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    image = cv2.imread(img_path)
    new_img = cv2.resize(image, (224, 224))
    # interpreter.set_tensor(input_details[0]['index'], [new_img])


    interpreter.invoke()

    rects = interpreter.get_tensor(output_details[0]['index'])

    print(rects)
    # scores = interpreter.get_tensor(output_details[2]['index'])
    #
    # for index, score in enumerate(scores[0]):
    #     if score > 0.5:
    #         draw_rect(image, rects[0][index])

    cv2.imshow("image", image)

    cv2.waitKey(0)



def main():
    apply_model_to_img("reptiles/dataset/chameleon/chameleon-424291__340.jpg")

if __name__ == "__main__":
    main()
