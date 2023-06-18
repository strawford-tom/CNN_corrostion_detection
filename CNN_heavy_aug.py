
# Import Libraries
import os
from keras.preprocessing.image import ImageDataGenerator
import sys
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from config import Config

print("Libraries Imported")

Config.IMG_SIZE = 128
Config.BATCH_SIZE = 64
Config.TRAINING_SIZE = 10551
Config.VALIDATION_SIZE = 5640
#class_weight = {0: 3.6146969696969697, 1: 0.5802646300530233}
#class_weight = {0:400, 1:0.8}
Config.CLASS_WEIGHT = {0:80, 1:1}
Config.NBEPOCHS = 150
#lr = 0.0000353
Config.LR = 0.000353
#lr = 0.00062729

def get_augmented_train_data(path):
        train_datagen = ImageDataGenerator(
                rescale=1./255,rotation_range=20, width_shift_range=0.06,
                             shear_range=0.25, zoom_range=0.5, 
                             horizontal_flip=True, vertical_flip=True)

        train_generator = train_datagen.flow_from_directory(
                path + '/Training_data_for_tiles',
                target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
                batch_size=Config.BATCH_SIZE,
                class_mode='binary')
        return train_generator

def get_train_data(path):
        train_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                path + '/Training_data_for_tiles',
                target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
                batch_size=Config.BATCH_SIZE,
                class_mode='binary')
        return train_generator


def get_validation_data(path):
        validation_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = validation_datagen.flow_from_directory(
                path + '/Validation_data_for_tiles',
                target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
                batch_size=Config.BATCH_SIZE,
                class_mode='binary')
        return validation_generator

# define cnn model
def define_model():
    model = Sequential()

    # first conv block
    model.add(Conv2D(128, (3, 3), activation = LeakyReLU(), kernel_initializer='he_uniform', padding='same',
                     input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3)))
    #model.add(Conv2D(96, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(SpatialDropout2D(0.2))

    # second conv block
    model.add(Conv2D(224, (3, 3), activation=LeakyReLU(), kernel_initializer='he_uniform', padding='same'))
    #model.add(Conv2D(96, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(SpatialDropout2D(0.1))

    # third conv block
    model.add(Conv2D(160, (3, 3), activation=LeakyReLU(), kernel_initializer='he_uniform', padding='same'))
    #model.add(Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(SpatialDropout2D(0.2))


    # forth conv block
    model.add(Conv2D(256, (3, 3), activation=LeakyReLU(), kernel_initializer='he_uniform', padding='same'))
    #model.add(Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(SpatialDropout2D(0.1))

    # flatteren 
    model.add(Flatten())
 
    model.add(Dense(60, activation=LeakyReLU(), kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot lossmode
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='validation')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='validation')
    pyplot.legend()
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename[:-3] + '_plot.png')
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    path = os.getcwd()
    train_data = get_augmented_train_data(path)
    validation_data = get_validation_data(path)
    # define model
    model = define_model()

    callback = EarlyStopping(monitor='loss', patience=45)
    # start time and fit model
    start = time.time()
    history = model.fit(train_data, epochs=Config.NBEPOCHS, batch_size=Config.BATCH_SIZE, 
                        steps_per_epoch=Config.TRAINING_SIZE/Config.BATCH_SIZE, validation_data=validation_data, 
                        validation_steps=Config.VALIDATION_SIZE/Config.BATCH_SIZE, callbacks=[callback], class_weight=Config.CLASS_WEIGHT, verbose=1)
    end = time.time()
    print('Training time (mins):', (end - start) / 60)
    print('Training time (hrs):', (end - start) / (60*60))
    # learning curves
    summarize_diagnostics(history)
    return model

# entry point, run the test harness
model = run_test_harness()
model.save("model.h5")



path = os.getcwd()


y_true = np.load(path + '/y_test.npy', allow_pickle=True)
X_test = np.load(path + '/X_test.npy', allow_pickle=True)

print("test data Loaded")

#model = keras.models.load_model(model_path + '/model.h5')

print("Model loaded")


def evaluate_predcitions(model, X_test, y_true):
    print("begining predictions")
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)

    print("creating confusion matrix and classification report")
    # Create confusion matrix
    print(confusion_matrix(y_true, y_preds))  # estimated targets as returned by a classifier

    print(classification_report(y_true, y_preds))


evaluate_predcitions(model, X_test, y_true)





