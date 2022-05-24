import tensorflow as tf 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from Unet import UnetModel
from dataLoader import *



dataPath = "E:/okulitu/itüYükseknew/2022bahar/DeepLearning/projects/MyCode/data/sequences/00"
dataimporter= dataLoader(dataPath)

# load Data Paths and create generator for our model
points, labels = dataimporter.loadDataPaths()
X_train, X_valid, y_train, y_valid = train_test_split(points, labels, test_size=0.2, random_state=42)

trainbatch = squenceDataGenerator(X_train,y_train,2)
validbatch = squenceDataGenerator(X_valid,y_valid,2)

#defining the model
unetModel = UnetModel()
unet = unetModel.UNetArch(input_size=(480,368,3), n_filters=64, n_classes=trainbatch.numberOfClasses)

## Create checkpoint call back to save checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Complie the model
unet.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

results = unet.fit(trainbatch,  epochs=1, validation_data=(validbatch),callbacks=callbacks_list)

import pandas as pd 
from matplotlib import pyplot as plt

pd.DataFrame(results.history).plot(figsize=(10,7), xlabel="epochs")
plt.title("Model_8 training curves")
plt.show()