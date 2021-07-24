import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_PATH = '/content/Music-Genre-Classification/data.json'

def load_data(dataset_path):
  """
  Utility to load input data and labels from json file.
  Returns a tuple having input data and labels.
  """
  with open(dataset_path, 'r') as fp:
    data = json.load(fp)

  # convert lists into numpy arrays
  inputs = np.array(data['mfcc'])
  labels = np.array(data['labels'])

  return inputs, labels

def build_model(inputs):
  """
  Build a model from the inputs.
  Returns a keras.Sequential model.
  """
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(inputs.shape[1],inputs.shape[2])),
    
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(rate=0.2),
    
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(rate=0.2),
    
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(rate=0.2),

    keras.layers.Dense(10, activation='softmax')
  ])
  return model

def plot_history(history):
  """
  Utility to plot the error and accuracy graph for the trained model.
  """
  fig, ax = plt.subplots(2, 1, figsize=(16,10), sharex=True)
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss =history.history['loss']
  val_loss = history.history['val_loss']

  # accuracy subplot
  ax[0].plot(acc, label='Train accuracy')
  ax[0].plot(val_acc, label='Test accuracy')
  ax[0].set_ylabel('Accuracy')
  ax[0].legend(loc='lower right')
  ax[0].set_title('Accuracy eval')

  # loss subplot
  ax[1].plot(loss, label='Train loss')
  ax[1].plot(val_loss, label='Test loss')
  ax[1].set_ylabel('Loss')
  ax[1].legend(loc='upper right')
  ax[1].set_title('Loss eval')
  ax[1].set_xlabel('Epochs')
  
  plt.show()


if __name__=='__main__':
  print('Loading data...')
  inputs, labels = load_data(DATASET_PATH)
  print('Loaded!')
  # split into train, test
  input_train, input_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.3)
  
  # print(f'Input train: {input_train.shape}')
  # print(f'Input test: {input_test.shape}')
  # print(f'Label train: {labels_train.shape}')
  # print(f'Label test: {labels_test.shape}')

  # build the network architecture
  print('Building model... ', end='')
  model = build_model(inputs)
  print('Done!')
  
  # compile the network
  optimizer = keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # summarize the model
  # print(model.summary())

  # train the model
  history = model.fit(
    input_train,labels_train,
    validation_data=(input_test, labels_test),
    epochs=50, batch_size=32,
    verbose=2
  )
  
  plot_history(history)
  plt.show()
  
