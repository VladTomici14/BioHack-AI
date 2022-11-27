from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cv2


class Utils:
    def __init__(self):
        pass

    def center_image(self, image):
        (width, height) = image.shape[:2]
        if height < width:
            x = width // 2 - (height // 2)
            y = 0
            target_w = x + height
            target_h = height
        else:
            y = height // 2 - (width // 2)
            x = 0
            target_w = width
            target_h = y + width

        resized_image = image[x:target_w, y:target_h]

        return resized_image

    def evaluate_model(self, validation_generator, model):
        y_val = validation_generator.classes
        y_pred = model.predict(validation_generator)
        y_pred = np.argmax(y_pred, axis=1)
        print(classification_report(y_val, y_pred))


class Plotting:
    """
    This class contains functions that are used for creating plots of data.
    """

    def __init__(self, history):
        self.history = history

    def loss_plot(self, save_plot):
        """
        Drawing a plot for loss comparison during the training.
            :param save_plot: the condition if we want to save the plot or not
            :return: shows a plot and saves it if the :param save_plot: == True
        """

        plt.figure(figsize=(20, 8))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save_plot:
            plt.savefig(
                f"docs/loss_plot_{datetime.date.today()}_{datetime.datetime.today().hour}-{datetime.datetime.today().minute}-{datetime.datetime.today().second}.png")
        plt.show()

    def accuracy_plot(self, save_plot):
        """
        Drawing a plot for accuracy comparison during the training.
            :param save_plot: the condition if we want to save the plot or not
            :return: shows a plot and saves it if the :param save_plot: == True
        """

        plt.figure(figsize=(20, 8))
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save_plot:
            plt.savefig(
                f"docs/accuracy_plot_{datetime.date.today()}_{datetime.datetime.today().hour}-{datetime.datetime.today().minute}-{datetime.datetime.today().second}.png")
        plt.show()

