from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import tensorflow.keras as keras
import numpy as np
import json

DATASET_PATH = 'data.json'

def load_data(dataset_path):
  """
  Utility to load input data and labels from json file.
    :param dataset_path(str): Path to JSON file
    :return inputs(ndarray): Inputs
    :return labels(ndarray): Labels
  """
  with open(dataset_path, 'r') as fp:
    data = json.load(fp)

  # convert lists into numpy arrays
  inputs = np.array(data['mfcc'])
  labels = np.array(data['labels'])
  mapping = np.array(data['mapping'])

  return inputs, labels, mapping


def split_data(inputs, labels, test_split=0.25, val_split=0.25):
  """
    Utility to split data into train, validation and test sets.
    :param inputs(ndarray): Input data
    :param labels(ndarray): Input Labels
    :param test_split(float): Test split ratio, default 0.25
    :param val_split(float): Validation split ratio, default 0.25
    :return split(tuple): Train, validation and test split for input data and labels
  """
  # create train/test split
  X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=test_split)

  # create train/val split
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split)
  
  # reshape all samples
  X_train = X_train[..., np.newaxis] # 4D array (num_samples, 130, 13, 1)
  X_val = X_val[..., np.newaxis]
  X_test = X_test[..., np.newaxis]

  return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape):
  """
    Builds the CNN model.
    :param input_shape(tuple): Shape of the input data
    :return model(keras.models.Sequential)
  """
  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPool2D((3,3), strides=(2,2), padding='same'))
  model.add(BatchNormalization())

  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPool2D((3,3), strides=(2,2), padding='same'))
  model.add(BatchNormalization())

  model.add(Conv2D(32, (2, 2), activation='relu'))
  model.add(MaxPool2D((2, 2), strides=(2,2), padding='same'))
  model.add(BatchNormalization())

  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.2))
  
  model.add(Dense(10, activation='softmax'))
  
  return model


def predict(model, X, y, mapping):
  """
    Predict for the given sample
    :param model(keras.model.Sequential): Trained model
    :param X(ndarray): Input data
    :param y(ndarray): Input label
    :param mapping(ndarray): Used to map the index to prediction label
  """
  X = X[np.newaxis, ...]
  predictions = model.predict(X, y)  # 2D

  # extract index with max values
  y_pred = np.argmax(predictions, axis=1)

  # print the results
  print(f'\nActual: {mapping[y]}\nPredicted: {mapping[y_pred][0]}')



if __name__=='__main__':
  # load the data
  print('Loading data... ', end='')
  inputs, labels, mapping = load_data(DATASET_PATH)
  print('Loaded!')

  # split data
  X_train, X_val, X_test, y_train, y_val, y_test = split_data(inputs, labels)

  # build the model
  input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
  model = build_model(input_shape)

  # compile the model
  optimizer = keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(
    optimizer=optimizer, 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  # train the model
  model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    batch_size=32, epochs=50
  )

  # evaluate model on test data
  test_acc, test_loss = model.evaluate(X_test, y_test, verbose=0)
  print(f'\nTest Accuracy: {test_acc}\nTest Loss: {test_loss}')

  # make predictions on a sample
  predict(model, X_test[100], y_test[100], mapping)


