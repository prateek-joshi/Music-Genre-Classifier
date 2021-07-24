import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf

DATASET_PATH = '/content/Music-Genre-Classification/data.json'

def load_data(dataset_path):
  with open(dataset_path, 'r') as fp:
    data = json.load(fp)

  # convert lists into numpy arrays
  inputs = np.array(data['mfcc'])
  labels = np.array(data['labels'])

  return inputs, labels

def build_model(inputs):
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(inputs.shape[1],inputs.shape[2])),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
  ])
  return model

if __name__=='__main__':
  inputs, labels = load_data(DATASET_PATH)
  # split into train, test
  input_train, input_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.3)
  
  # print(f'Input train: {input_train.shape}')
  # print(f'Input test: {input_test.shape}')
  # print(f'Label train: {labels_train.shape}')
  # print(f'Label test: {labels_test.shape}')

  # build the network architecture
  model = build_model(inputs)
  
  # compile the network
  optimizer = keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # summarize the model
  print(model.summary())

  # train the model
  model.fit(
    input_train,labels_train,
    validation_data=(input_test, labels_test),
    epochs=50, batch_size=32
  )
  
  
