import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from dataPreprocess import dataPreprocess
from model import LSTModel, CNNModel
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import numpy as np
from sklearn.metrics import r2_score

st.title('Stock Movement Prediction Using LSTM and CNN Models')
# st.header("Any prediction made on this website is for academic purpose only.")
st.subheader('Powered by Streamlit and Alpha Vantage API.')

alpha = TimeSeries(key='C1973OKKIG1XB6KF')
searchTicker = st.text_input('Search for company. You can enter either a name or a symbol. For example: "AAPL" is the symbol and "Apple Inc." is the name', max_chars=100)
# create empty dictionary to store tickers and their symbols
matches = {}

if(len(searchTicker) != 0):
  # this will return a tuple with all of the data in index 0
  repsonses = alpha.get_symbol_search(searchTicker)
  # get the data in index 0 -> this data is a list which has a dictionary at each index
  extract = repsonses[0]

  # loop through the list to get the ticker and symbol
  # only need the symbol and the company's name
  for data in extract:
    for key, value in data.items():
      if key == '1. symbol':
        k = value
      if key == '2. name':
        v = value
        matches[k] = v  # adding a key-value pair to the dictionary

def formatOutput(options):
  return matches[options]

#st.write('If you want to change the values of sliders, please do so before searching for a company')
selectedTicker = st.selectbox('Select your company', options=list(matches.keys()), format_func=formatOutput)

if selectedTicker != None:
  selectedSplit = st.sidebar.slider('Choose a spilt ratio to split the data set into input and output set', min_value=0.1, max_value=0.99, value=0.85)
  selectedTimeSteps = st.sidebar.slider('Choose a number for the time step', min_value=10, max_value=100, value=50)
  selectedBatchSize = st.sidebar.slider('Choose a number for the batch size', min_value=10, max_value=100, value=32)
  selectedEpochNum = st.sidebar.slider('Choose the number of epoch', min_value=1, max_value=100, value=10)
  selectedKFold = st.sidebar.checkbox('Do you want to use K-Fold cross-validation?')
  selectedFeatures = st.sidebar.multiselect('Select one or more features', options=['Close', 'Volume', 'Open', 'High', 'Low'], default=['Close'])

  if selectedSplit != None or selectedTimeSteps != None or selectedBatchSize != None or selectedEpochNum != None:
    if st.button('Predict the stock trend'):
      with st.spinner('Please wait a moment...'):
        # get the data and preprocess it
        data = dataPreprocess(selectedTicker, selectedSplit, selectedKFold, featuresIncluded=selectedFeatures)

        # build the models
        modelLSTM = LSTModel()
        modelLSTM.createModel(100, selectedTimeSteps, len(selectedFeatures), 0.2)

        modelCNN = CNNModel()
        modelCNN.createModel(selectedTimeSteps, len(selectedFeatures))

        if selectedKFold:
          allTestLSTM = []
          allTestCNN = []
          x, y = data.getData(selectedTimeSteps)
          kf = KFold(n_splits=10)

          for trainIndex, testIndex in kf.split(x, y):
            xTrain = x[trainIndex]
            yTrain = y[trainIndex]
            lookback = math.floor((len(xTrain) - selectedTimeSteps) / selectedBatchSize)
            modelLSTM.train(x=x, y=y, epochNum=selectedEpochNum, batchSize=selectedBatchSize, lookback=lookback, path='savedModel')
            modelCNN.train(x=x, y=y, epochNum=selectedEpochNum, batchSize=selectedBatchSize, lookback=lookback, path='saveModel')
            testLSTM = modelLSTM.getMAE(x[testIndex], y[testIndex])
            testCNN = modelCNN.getMAE(x[testIndex], y[testIndex])
            allTestLSTM.append(testLSTM)
            allTestCNN.append(testCNN)

          maeLSTM = np.mean(allTestLSTM)
          maeCNN = np.mean(allTestCNN)

          x, y = data.getData(selectedTimeSteps)
          predictionsLSTM = modelLSTM.predictOneDayAhead(x)
          predictionsCNN = modelCNN.predictOneDayAhead(x)

          if len(predictionsLSTM) > 0 and len(predictionsCNN) > 0:
            st.success('Finished processing! Here is the predictions')
            st.write("LSTM Model results:")
            st.write("Average K-Fold cross-validation loss for LSTM: ", maeLSTM)
            st.write("The predicted log return is:", predictionsLSTM[-1])

            # calculate the mean square error
            mae = modelLSTM.getMAE(x, y)
            st.write("Mean Absolute Error:", mae)
            # calculate the R-squared score
            scoreL = r2_score(y, predictionsLSTM)
            st.write("R-squared:", scoreL)

            predictedPriceChangeL = np.exp(predictionsLSTM[-1]) - 1
            st.write("Converting it to a price return or a price change:", predictedPriceChangeL)

            st.write("CNN Model results:")
            st.write("Average repeated K-Fold cross-validation loss for CNN: ", maeCNN)
            st.write("The predicted log return is:", predictionsCNN[-1])

            # calculate the mean square error
            mae = modelCNN.getMAE(x, y)
            st.write("Mean Absolute Error:", mae)
            # calculate the R-squared score
            scoreC = r2_score(y, predictionsCNN)
            st.write("R-squared:", scoreC)

            predictedPriceChangeC = np.exp(predictionsCNN[-1]) - 1
            st.write("Converting it to a price return or a price change:", predictedPriceChangeC)

        else:
          # train the models
          x, y = data.getTrainSet(selectedTimeSteps)
          lookback = math.floor((data.trainLength - selectedTimeSteps) / selectedBatchSize)
          modelLSTM.train(x=x, y=y, epochNum=selectedEpochNum, batchSize=selectedBatchSize, lookback=lookback, path='savedModel')
          modelCNN.train(x=x, y=y, epochNum=selectedEpochNum, batchSize=selectedBatchSize, lookback=lookback, path='saveModel')

          # test the model
          xTest, yTest = data.getTestSet(selectedTimeSteps)
          testLSTM = modelLSTM.predictOneDayAhead(xTest)
          testCNN = modelCNN.predictOneDayAhead(xTest)

          x, y = data.getData(selectedTimeSteps)
          predictionsLSTM = modelLSTM.predictOneDayAhead(x)
          predictionsCNN = modelCNN.predictOneDayAhead(x)

          if len(predictionsLSTM) > 0 and len(predictionsCNN) > 0:
            st.success('Finished processing! Here is the predictions')
            st.write("LSTM Model results:")
            st.write("The predicted log return is:", predictionsLSTM[-1])

            # calculate the mean square error
            mae = modelLSTM.getMAE(x, y)
            st.write("Mean Absolute Error:", mae)
            # calculate the R-squared score
            scoreL = r2_score(y, predictionsLSTM)
            st.write("R-squared:", scoreL)

            predictedPriceChangeL = np.exp(predictionsLSTM[-1]) - 1
            st.write("Converting it to a price return or a price change:", predictedPriceChangeL)

            st.write("CNN Model results:")
            st.write("The predicted log return is:", predictionsCNN[-1])

            # calculate the mean square error
            mae = modelCNN.getMAE(x, y)
            st.write("Mean Absolute Error:", mae)
            # calculate the R-squared score
            scoreC = r2_score(y, predictionsCNN)
            st.write("R-squared:", scoreC)

            predictedPriceChangeC = np.exp(predictionsCNN[-1]) - 1
            st.write("Converting it to a price return or a price change:", predictedPriceChangeC)

          # illustrations
          figL = plt.figure(facecolor='white')
          ax = figL.add_subplot(111)
          ax.plot(yTest, label="True Log Return")
          plt.plot(testLSTM, label="Predicted Log Return")
          plt.ylabel('Log Returns')
          plt.xlabel('Epoch')
          plt.legend()
          figL.suptitle('LSTM Test Results')
          st.pyplot(figL)

          figC = plt.figure(facecolor='white')
          ax = figC.add_subplot(111)
          ax.plot(yTest, label="True Log Return")
          plt.plot(testCNN, label="Predicted Log Return")
          plt.ylabel('Log Returns')
          plt.xlabel('Epoch')
          plt.legend()
          figC.suptitle('CNN Test Results')
          st.pyplot(figC)




