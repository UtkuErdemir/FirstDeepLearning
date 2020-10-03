import keras.datasets as datasets
import keras.layers as layers
import keras.models as models
import keras.utils as utils
import numpy as np
import matplotlib.pyplot as plt

dataset = datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = utils.normalize(x_train, axis=1)
x_test = utils.normalize(x_test, axis=1)
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 3)
val_loss, val_accuracy = model.evaluate(x_test, y_test)
f"loss: {val_loss} accuracy: {val_accuracy}"
predicts = model.predict(x_test)
print(np.argmax(predicts[5]))
plt.imshow(x_test[5],cmap = plt.cm.binary)
plt.show()