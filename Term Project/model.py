import dataPreprocess
import datetime as date
import numpy as np
import os
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM, InputLayer, Conv1D, AveragePooling1D, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class LSTModel():
  '''This class creates the LSTM model, train it with given training set, and make a prediction.'''

  def __init__(self):
    '''Initialize the model.'''
    self.model = Sequential()

  def getModel(self, path):
    self.model = load_model(path)

  def createModel(self, LSTMNeurons, timesteps, dimension, dropout):

    #add layers to the model
    self.model.add(LSTM(LSTMNeurons, input_shape=(timesteps - 1, dimension), return_sequences=True))
    self.model.add(Dropout(dropout))
    self.model.add(LSTM(LSTMNeurons, return_sequences=True))
    self.model.add(LSTM(LSTMNeurons, return_sequences=False))
    self.model.add(Dense(1, activation="linear"))

    self.model.compile(loss='mae', optimizer='adam')
    return self.model

  def train(self, x, y, epochNum, batchSize, lookback, path):
    dateNow = date.datetime.now().strftime("%m-%d-%Y")
    modelName = "LSTM: " + dateNow + " number-of-epochs:" + str(epochNum)
    savePath = os.path.join(path, modelName) + ".h5"
    callbacks = [ModelCheckpoint(filepath=savePath, monitor='val_loss', save_best_only=True)]
    self.model.fit(x=x, y=y, validation_split=0.1, steps_per_epoch=lookback, epochs=epochNum, callbacks=callbacks)

  def predictOneDayAhead(self, dataSet):
    predicted = self.model.predict(dataSet)
    predicted = np.reshape(predicted, (predicted.size))
    return predicted

  def getMAE(self, x, y):
    mae = self.model.evaluate(x, y)
    return mae

  def getHistoryLoss(self):
    history = self.model.history['loss']
    return history

  def getValLoss(self):
    history = self.model.history['val_loss']
    return history

class CNNModel():
  def __init__(self):
    '''Initialize the model'''
    self.model = Sequential()

  def createModel(self, timesteps, dimension):
    self.model.add(InputLayer(input_shape=(timesteps - 1, dimension)))
    self.model.add(Conv1D(kernel_size=2, filters=128, strides=1, use_bias=True, activation='relu', kernel_initializer='VarianceScaling'))
    self.model.add(AveragePooling1D(pool_size=2, strides=1))
    self.model.add(Conv1D(kernel_size=2, filters=64, strides=1, use_bias=True, activation='relu', kernel_initializer='VarianceScaling'))
    self.model.add(AveragePooling1D(pool_size=2, strides=1))
    self.model.add(Flatten())
    self.model.add(Dense(1, activation='linear', kernel_initializer='VarianceScaling'))

    self.model.compile(loss='mae', optimizer='adam')

  def train(self, x, y, epochNum, batchSize, lookback, path):
    dateNow = date.datetime.now().strftime("%m-%d-%Y")
    modelName = "CNN: " + dateNow + " number-of-epochs:" + str(epochNum)
    savePath = os.path.join(path, modelName) + ".h5"
    callbacks = [ModelCheckpoint(filepath=savePath, monitor='val_loss', save_best_only=True)]
    self.model.fit(x=x, y=y, validation_split=0.1, steps_per_epoch=lookback, epochs=epochNum, callbacks=callbacks)

  def predictOneDayAhead(self, dataSet):
    predicted = self.model.predict(dataSet)
    predicted = np.reshape(predicted, (predicted.size))
    return predicted

  def getMAE(self, x, y):
    mae = self.model.evaluate(x, y)
    return mae

  def getHistoryLoss(self):
    history = self.model.history
    return history

  def getValLoss(self):
    history = self.model.history['val_loss']
    return history

