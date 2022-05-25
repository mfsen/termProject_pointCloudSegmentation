from tensorflow.keras.models import load_model
import tensorflow as tf
from dataLoader import *
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = load_model('weights-improvement-08-0.99.hdf5')
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


dataPath = "E:/okulitu/itüYükseknew/2022bahar/DeepLearning/projects/MyCode/data/sequences/00"
dataimporter= dataLoader(dataPath)

# load Data Paths and creamte generator for our model
points, labels = dataimporter.loadDataPaths()

preprocess = dataPreProcessor()
xtest, ytest = preprocess.processBanchOfData([points[3]], [labels[3]])

result = model.predict(xtest)
pred_mask = tf.argmax(result[0], axis=-1)
pred_mask = pred_mask[..., tf.newaxis]

labelresult = np.array(pred_mask[:,:,0])
labelresult = np.expand_dims(labelresult,axis=2)


cm = confusion_matrix(ytest[0].flatten(), labelresult.flatten())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

accuracyperclass = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(accuracyperclass.diagonal())

for i in range(len(preprocess.uniqueClasses)):
    labelresult  = np.where(labelresult == i, [preprocess.SemKITTI_label_name[int(preprocess.uniqueClasses[i])]  ], labelresult)
f = preprocess.readPoints([points[3]], [labels[3]],labelresult)


np.savetxt('test.txt', f[0], delimiter=',')
# fill = open("testpoints.txt","w")
# for i in f:
#     fill.write(",".join([str(x) for x in list(i)])+"\n")
# fill.close()
import cv2  
cv2.imwrite("sdsd.jpeg",labelresult)

