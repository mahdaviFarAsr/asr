import tensorflow as tf
from tensorflow import  keras
import pandas as pd
import matplotlib.pyplot as plt


print("tensorflow is version: ", tf.__version__)
print('keras version: ', keras.__version__)

# import dataset
fashion_mnist = keras.datasets.fashion_mnist
(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()

# view dataset
print('xTrain_shape: ' , xTrain.shape, 'xTrain_dType: ', xTrain.dtype)
print('yTrain_shape: ', yTrain.shape, 'yTrain_dType: ', yTrain.dtype)
print('******************')
print('xTest_shape: ',xTest.shape, 'xTest_dType: ', xTest.dtype)
print('yTest_shape: ', yTest.shape, 'yTest_dType: ', yTest.dtype)

# create validation dataset and normalizeing
xValid, xTrain = xTrain[50000:]/255.0, xTrain[:50000]/255.0
yValid, yTrain = yTrain[50000:] , yTrain[:50000]

# create sequential model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# train model
history = model.fit(xTrain,yTrain,epochs=30, validation_data=(xValid, yValid))

# diagram
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()