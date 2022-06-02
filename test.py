from tensorflow.keras.models import load_model
import tensorflow as tf
from dataLoader import *
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# loading model that is saved while training
model = load_model('weights-improvement-08-0.99.hdf5')
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#write your data path here
dataPath = " "
dataimporter= dataLoader(dataPath)

# load Data Paths and creamte generator for our model
points, labels = dataimporter.loadDataPaths()

#Preprocessing input data
preprocess = dataPreProcessor()
xtest, ytest = preprocess.processBanchOfData([points[3]], [labels[3]])

result = model.predict(xtest)
pred_mask = tf.argmax(result[0], axis=-1)
pred_mask = pred_mask[..., tf.newaxis]

labelresult = np.array(pred_mask[:,:,0])
labelresult = np.expand_dims(labelresult,axis=2)

# ploting confusion matrix
cm = confusion_matrix(ytest[0].flatten(), labelresult.flatten())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#calculating accuracy per class
accuracyperclass = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(accuracyperclass.diagonal().flatten())

for i in range(len(preprocess.uniqueClasses)):
    labelresult  = np.where(labelresult == i, [preprocess.SemKITTI_label_color[int(preprocess.uniqueClasses[i])]  ], labelresult)

f = preprocess.readPoints([points[3]], [labels[3]],labelresult)

#saving segmented points
np.savetxt('test.txt', f[0], delimiter=',')

#printing bev image
import cv2  
cv2.imwrite("bevImage.jpeg",labelresult)

