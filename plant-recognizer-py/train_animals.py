from tqdm import tqdm_notebook as tgdm
import tensorflow as tf
import numpy as np
import warnings
import argparse
import sklearn
import pandas
import cv2
import os

from utils import Utils

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.utils import to_categorical
import keras.applications

warnings.simplefilter(action='ignore', category=FutureWarning)

class Trainer:
    def __init__(self):
        self.animal_classes = []
        self.files = []
        self.categories = []
        self.utils = Utils()

    def prepare_data(self, dir_path):
        """
        Preparing the dataset before training.

            :param dir_path: the path of the dataset directory

            :return:
        """

        # ------- reading the classes labels from the dataset -----
        for x in os.listdir(dir_path):
            if os.path.isdir(f"{dir_path}/{x}"):
                self.animal_classes.append(x)

        # --------- reading the files from the dataset -------
        for k, folder in enumerate(self.animal_classes):
            filenames = os.listdir(f"{dir_path}/{folder}")
            for file in filenames:
                self.files.append(f"{dir_path}/{folder}/{file}")
                self.categories.append(k)

        data = pandas.DataFrame({
            "filename": self.files,
            "category": self.categories
        })

        train_data = pandas.DataFrame(columns=["filename", "category"])
        for i in range(len(self.animal_classes)):
            train_data = train_data.append(data[data.category == i].iloc[:500, :])

        y = train_data["category"]
        x = train_data["filename"]
        y = train_data["category"]

        train_data = train_data.reset_index(drop=True)

        x, y = sklearn.utils.shuffle(x, y, random_state=8)

        # ---------- resizing the images ------------
        images = []
        for i, file_path in enumerate(train_data.filename.values):
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image.shape[0] > image.shape[1]:
                tile_size = (int(image.shape[1] * 256 / image.shape[0]), 256)
            else:
                tile_size = (256, int(image.shape[0] * 256 / image.shape[1]))

            image = self.utils.centering_image(cv2.resize(image, dsize=tile_size))

            image = image[16:240, 16:240]
            images.append(image)

        # ---------- adding the images to a numpy array ----------
        images = np.array(images)

        # --------- preparing the arrays ----------
        data_num = len(y)
        random_index = np.random.permutation(data_num)

        x_shuffle = []
        y_shuffle = []

        print(f"y: {y}")

        for i in range(data_num):
            x_shuffle.append(images[random_index[i]])
            # y_shuffle.append(y[random_index[i]])
            rand_ind = int(random_index[i])
            print(type(y[int(random_index[i])]))

        x = np.array(x_shuffle)
        y = np.array(y_shuffle)
        val_split_num = int(round(0.2 * len(y)))
        x_train = x[val_split_num:]
        y_train = y[val_split_num:]
        x_test = x[:val_split_num]
        y_test = y[:val_split_num]

        print('x_train', x_train.shape)
        print('y_train', y_train.shape)
        print('x_test', x_test.shape)
        print('y_test', y_test.shape)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    def create_model(self):
        img_rows, img_cols, img_chnls = 224, 224, 3

        base_model = keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(img_rows, img_cols, img_chnls))

        add_model = Sequential()
        add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        add_model.add(Dense(256, activation="relu"))
        add_model.add(Dense(10, activation="softmax"))

        model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9, metrics=["accuracy"]))

        # TODO: generate the model architecture
        model.summary()

    def train(self, dir_path):
        self.prepare_data(dir_path=dir_path)

        # self.create_model()


def main():
    # -------- configuring the argparser --------
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, default="animals-10/raw-img", help="path of the dataset")
    args = ap.parse_args()

    Trainer().train("animals-10/raw-img")


if __name__ == "__main__":
    main()
