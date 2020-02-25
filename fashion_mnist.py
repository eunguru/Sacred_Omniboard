import tensorflow as tf
from tensorflow import keras
import numpy as np

num_epochs = 5
hidden_size = 128
num_classes = 10

# data load
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''
print("type(train_images): {}".format(type(train_images)))
print("train_images.shape: {}, train_labels.shape: {}".format(train_images.shape, train_labels.shape))
print("train_images.shape: {}, train_labels.shape: {}".format(test_images.shape, test_images.shape))
'''

class_names = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# data preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
                            keras.layers.Flatten(input_shape = (28, 28)),
                            keras.layers.Dense(hidden_size, activation='relu'),
                            keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.summary()



# callback: 
cp_callback = keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: print(logs)
)

# training        
model.fit(train_images, train_labels, epochs=num_epochs,
            validation_data = (test_images, test_labels),
            callbacks = [cp_callback])

# evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("test acc: {]", test_acc)

predictions = model.predict(test_images)
predictions[0]

np.argmax(predictions[0])
test_labels[0]