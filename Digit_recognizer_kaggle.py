# Import external libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import h5py


#-> Import internal libs
from helper_functions import DataVisualization
from Models import ANN_1FC_10Cls


# Prevent Memmory overflow in GPU
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def import_data():
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv').to_numpy()

    y_train = train_set.label.to_numpy()
    x_train = train_set.drop(['label'],axis=1).to_numpy()

    x_train = x_train.reshape(x_train.shape[0],28,28)
    x_test = test_set.reshape(test_set.shape[0],28,28)

    return x_train,y_train,x_test

def main():
    x_train, y_train, x_test = import_data()

    # it hsould be 28,28,1 not 28 ,28 for vgg mini
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05)
    DataVisualization.print_data_statistics(x_train, x_val, x_test, y_train, y_val)
    # Normalize input features
    x_train, x_val, x_test = x_train / 255.0, x_val / 255, x_test / 255.0
    class_names = [str(i) for i in range(10)]
    #DataVisualization.show_image_plt("Training data set example", x_train, y_train, class_names, max=1)
    '''
    # ///////////////////// Define the architecture ////////////////////
    model = tf.keras.models.Sequential()
    # First CONV => RELU => CONV => RELU => POOL layer set
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu',input_shape=(28,28,1)))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    # Second CONV => RELU => CONV => RELU => POOL layer set
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    # first (and only) set of FC => RELU layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    # softmax classifier
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    # ///////////////////// Train the Model ////////////////////
    # process = model.fit(x_train, y_train, epochs=5) # Simple model


    history = model.fit(x_train, y_train, batch_size=64, epochs=15, validation_data=(x_val, y_val))
    #DataVisualization.show_learning_status(history)
    model.save('my_model.h5')
    '''
    # ///////////////////// Touch Test set :Evaluate the model  ////////////////////
    #test_loss, test_acc = model.evaluate(x_val, y_val)
    #print('\nTest accuracy: {} , Test loss : {}'.format(test_acc, test_loss))
    from tensorflow.keras.models import load_model
    # load model
    model = load_model('my_model.h5')
    # /// Visualize first 25 test data with its classes
    predicted = np.argmax(model.predict(x_test), axis=1)
    submission = pd.read_csv('sample_submission.csv')
    submission['Label'] = predicted
    submission.to_csv('submission.csv', index=False)



if __name__ =='__main__':
    main()