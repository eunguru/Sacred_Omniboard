import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.randomness import get_seed

ex = Experiment('fashion_mnist')
ex.observers.append(MongoObserver.create(url='localhost:27017',
                                        db_name='sacred_omniboard'))

@ex.config
def get_config():
    num_epochs = 5
    hidden_size = 128
    num_classes = 10

@ex.capture
def set_model_path(seed):
    model_dir = os.getcwd()
    model_filename = 'model/my_model_{}.h5'.format(seed)
    model_path = os.path.join(model_dir, model_filename)
    
    return model_path

def write_logs(ex, logs):
    #print(logs)
    ex.log_scalar('loss', logs.get('loss'))
    ex.log_scalar('val_loss', logs.get('val_loss'))
    ex.log_scalar('accuracy', logs.get('accuracy'))
    ex.log_scalar('val_accuracy', logs.get('val_accuracy'))

@ex.capture
def create_model(hidden_size, num_classes):
    # create model
    model = keras.Sequential([
                                keras.layers.Flatten(input_shape = (28, 28)),
                                keras.layers.Dense(hidden_size, activation='relu'),
                                keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.summary()

    return model

@ex.capture
def train(num_epochs):
    print(tf.__version__)

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
    
    # create model
    model = create_model()

    # callback
    cp_callback = keras.callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, logs: write_logs(ex, logs)
    )

    # training        
    model.fit(train_images, train_labels, epochs=num_epochs,
                validation_data = (test_images, test_labels),
                callbacks = [cp_callback])

    # model save
    model_path = set_model_path()
    model.save(model_path)

    # add artifact: content_type - mimetype
    ex.add_artifact(filename=model_path, content_type='application/x-hdf5')

    # evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("test_loss: {}, test acc: {}".format(test_loss, test_acc))

@ex.automain
def run():
    train()