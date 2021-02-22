import tensorflow as tf
from tensorflow import keras
from GoogLeNet import GoogLeNet
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
num_classes = len(np.unique(y_train))
print(num_classes)

model = GoogLeNet.create_model((224, 224, 3), num_classes)
print(model)
#model.compile(optimizer='adam', loss='categorical_crossentropy')
#model.fit(x_train, y_train)
