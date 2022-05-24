from tensorflow.keras.models import load_model
import tensorflow as tf
from dataLoader import *

model = load_model('weights-improvement-08-0.99.hdf5')
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

dataPath = "E:/okulitu/itüYükseknew/2022bahar/DeepLearning/projects/MyCode/data/sequences/00"
dataimporter= dataLoader(dataPath)

# load Data Paths and create generator for our model
points, labels = dataimporter.loadDataPaths()

preprocess = dataPreProcessor()
xtest, ytest = preprocess.processBanchOfData([points[3]], [labels[3]])

result = model.predict(xtest)
pred_mask = tf.argmax(result[0], axis=-1)
pred_mask = pred_mask[..., tf.newaxis]

labelresult = np.array(pred_mask[:,:,0])
labelresult = np.expand_dims(labelresult,axis=2)
for i in range(len(preprocess.uniqueClasses)):
    labelresult  = np.where(labelresult == i, [preprocess.SemKITTI_label_name[int(preprocess.uniqueClasses[i])]  ], labelresult)

import cv2  
cv2.imwrite("sdsd.jpeg",labelresult)

