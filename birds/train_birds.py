# import keras.applications.mobilenet_v2
# import numpy as np
# import pandas
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
#
# import keras
# from keras.models import Model
# from keras.layers import Dense, Flatten, Conv2D, Dropout, Resizing, Rescaling
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras_preprocessing.image import ImageDataGenerator
#
# from pathlib import Path
# import os.path
#
#
# from helper_functions import *
#
#
#
# batch_size = 32
# image_size = (300, 300)
#
# dataset_path = "dataset/train"
#
# image_dir = Path(dataset_path)
#
# filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
#
# labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
#
# filepaths = pandas.Series(filepaths, name="Filepath").astype(str)
# labels = pandas.Series(labels, name="Label")
#
# image_df = pandas.concat([filepaths, labels], axis=1)
#
#
# # ======== visualizing some of the images from the dataset into a plot =========
# figure, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), subplot_kw={"xticks": [], "yticks": []})
#
# random_index = np.random.randint(0, len(image_df), 16)
# for i, ax in enumerate(axes.flat):
#     ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
#     ax.set_title(image_df.Label[random_index[i]])
#
# plt.tight_layout()
# plt.show()
#
# # ========= separating training / testing data ===========
# train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)
#
# train_generator = ImageDataGenerator(
#     preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
#     validation_split=0.2
# )
#
# test_generator = ImageDataGenerator(
#     preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
# )
#
# train_images = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col="Filepath",
#     y_col="Label",
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=True,
#     seed=42,
#     subset='training'
# )
#
# val_images = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=True,
#     seed=42,
#     subset='validation'
# )
#
# test_images = test_generator.flow_from_dataframe(
#     dataframe=test_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=False
# )
#
# resize_and_rescale = keras.Sequential([
#     Resizing(224, 224),
#     Rescaling(1./255)
# ])
#
# pretrained_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )
#
# pretrained_model.trainable = False
#
# # Create checkpoint callback
# checkpoint_path = "birds_classification_model_checkpoint"
# checkpoint_callback = ModelCheckpoint(checkpoint_path,
#                                       save_weights_only=True,
#                                       monitor="val_accuracy",
#                                       save_best_only=True)
#
# # Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
# early_stopping = EarlyStopping(monitor="val_loss", # watch the val loss metric
#                                patience=5,
#                                restore_best_weights=True) # if val loss decreases for 3 epochs in a row, stop training
#
# # ====== training the model =========
# inputs = pretrained_model.input
# x = resize_and_rescale(inputs)
#
# x = Dense(256, activation='relu')(pretrained_model.output)
# x = Dropout(0.2)(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.2)(x)
# outputs = Dense(400, activation='softmax')(x)
#
# model = Model(inputs=inputs, outputs=outputs)
#
# model.compile(
#     optimizer=keras.optimizers.Adam(0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# history = model.fit(
#     train_images,
#     steps_per_epoch=len(train_images),
#     validation_data=val_images,
#     validation_steps=len(val_images),
#     epochs=100,
#     callbacks=[
#         early_stopping,
#         create_tensorboard_callback("training_logs", "bird_classification"),
#         checkpoint_callback
#     ]
# )
#
