import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow import keras


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


num_classes = 10


train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)
print("train images shape:", train_images.shape)
print(train_images.shape[0], "train samples")
print(test_images.shape[0], "test samples")

train_images = keras.utils.to_categorical(test_images, num_classes)
train_labels = keras.utils.to_categorical(train_labels, num_classes)


#train_images, test_images = train_images/255.0, test_images/255.0

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


input_shape = (50000, 32, 3)
model = models.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))
model.summary()
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history =model.fit(train_images, train_labels, batch_size=128, epochs=15)
score = model.evaluate(test_images, test_labels, verbose=0)






