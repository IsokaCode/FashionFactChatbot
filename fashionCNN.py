# Upated NN to a simple CNN which will achive a 91% test accuracy on fashion MNIST


# Helper Libraries
import os
import numpy as np
import matplotlib.pyplot as plt


# TensorFlow and tf.keras
import tensorflow as tf
print(tf.__version__)

# importing and loading the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# reshape the data
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# preprocess the data
train_images = train_images / 255.0 
test_images = test_images / 255.0

# building CNN model 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),# overfitting prevention
    tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation= 'relu'),
    tf.keras.layers.Dense(10),
])  

# compiling model
model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

# training model
model.fit(train_images, train_labels, epochs=15)

# Evaluate model accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                        tf.keras.layers.Softmax()])
    
predictions = probability_model.predict(test_images)

# outputs the 10 label probabilities for the first (0) test image 
print(predictions[0])

# save model 
if os.path.isfile('/mnt/c/dev/uni/ai/fashion_CNN_model.h5') is False: # avoid overwriting the model
  model.save('fashion_CNN_model.h5')
