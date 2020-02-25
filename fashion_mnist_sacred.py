import tensorflow as tf
from tensorflow import keras
import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('fashion_mnist')
ex.observers.append(MongoObserver.create(url='localhost:27017',
                                        db_name='sacred_omniboard'))

@ex.config
def get_config():
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

# make model
@ex.caputure
def create_model(hidden_size, num_classes):    
    model = keras.Sequential([
                                keras.layers.Flatten(input_shape = (28, 28)),
                                keras.layers.Dense(hidden_size, activation='relu'),
                                keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.summry()
    return model

def write_logs():
    print(logs)
    ex.log_scalar('loss', lig)
@ex.capture
def train_keras():
    # callback
    cp_callback = keras.callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, logs: write_logs(ex, logs)
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

@ex.automain
def run():

