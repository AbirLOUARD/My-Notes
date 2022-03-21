import matplotlib.pyplot as plt
from tensorflow.keras import datasets
import tensorflow as tf

(training_images, training_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

training_images, test_images = training_images /255.0, test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation= "softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])
model.summary()

history = model.fit(training_images, training_labels, epochs=5,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')








