import warnings
import argparse

from utils.utils import Utils, Plotting

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.applications
import keras.utils

warnings.simplefilter(action='ignore', category=FutureWarning)


class Trainer:
    def __init__(self):
        self.animal_classes = []
        self.utils = Utils()

        self.batch_size = 32
        self.image_size = 224

    def create_model(self):
        base_model = keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(self.image_size, self.image_size, 3))

        # -------- adding layers to the model ---------
        last = base_model.layers[-2].output
        x = GlobalAveragePooling2D()(last)
        x = Dense(512, 'relu')(x)
        x = Dense(9, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

        # --------- plotting the model layers ---------
        keras.utils.plot_model(model, "docs/model_layers_reptiles.png", True)

        # ----------- compiling the model -----------
        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])

        model.summary()

        return model

    def return_callbacks(self):
        model_name = "model_reptiles.h5"

        checkpoint = ModelCheckpoint(model_name,
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only=True,
                                     verbose=1)

        early_stopping = EarlyStopping(monitor="val_loss",
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       restore_best_weights=True)

        learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.2,
                                                    min_lr=0.00000001)

        return [checkpoint, early_stopping, learning_rate_reduction]

    def train(self, dir_path):
        model = self.create_model()

        datagen = ImageDataGenerator(
            rescale=1 / 255.,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.15
        )

        train_generator = datagen.flow_from_directory(
            dir_path,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            shuffle=True,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            dir_path,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            shuffle=False,
            class_mode='categorical',
            subset='validation'
        )

        history = model.fit(train_generator,
                            epochs=60,
                            validation_data=validation_generator,
                            callbacks=self.return_callbacks())

        Plotting(history).loss_plot(save_plot=True)
        Plotting(history).accuracy_plot(save_plot=True)


def main():
    Trainer().train("reptiles/dataset")


if __name__ == "__main__":
    main()
