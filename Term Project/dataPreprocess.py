import datetime as dt
import urllib.request, json
import pandas_datareader
from pandas_datareader import data
import pandas as pd
import numpy as np

class dataPreprocess():
  '''This class makes API calls in order to get the data from Alpha Vantage, then it will
     preprocess the fetched data. '''
  data = []

  def __init__(self, ticker, splitRatio, kFold=False, featuresIncluded=['Close']):
    '''Make the API calls to fetch data and split data into test and training data.
       Parameters:
        ticker(string): the name of the company
        splitRatio(int): the ratio/percentage to split the dataset
        featuresIncluded(string array): features to be used, default is the close price'''

    # Make API calls.
    # Can get a free individual API key at https://www.alphavantage.co
    api_key = 'C1973OKKIG1XB6KF'
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    # Translate from JSON to DataFrame. Sort the data according to lastest date
    with urllib.request.urlopen(url_string) as url:
      data = json.loads(url.read().decode())
      # Extract the data from JSON object -> will be stored as a dictionary
      data = data['Time Series (Daily)']
      # Build the DataFrame
      df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Open', 'Close', 'Volume'])
      for a,b in data.items():
                # date = dt.datetime.strptime(a, '%Y-%m-%d')
                date = pd.to_datetime(a)  # easier way to transform to date object
                data_row = [date,float(b['3. low']), float(b['2. high']),
                            float(b['1. open']), float(b['4. close']), float(b['5. volume'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
    df = df.sort_values(by=['Date'], ascending=True) # sort the value by date

    if kFold == False:
      # Split data into training and test
      split = int(len(data) * splitRatio)
      self.trainData = df.get(featuresIncluded).values[:split]
      self.testData = df.get(featuresIncluded).values[split:]


      # Get the specified features
      features = featuresIncluded

      self.trainLength = len(self.trainData)
      self.testLength = len(self.testData)

    self.data = df.get(featuresIncluded).values

  def getTrainSet(self, time_step):
    '''Create x and y train data sequences.
       Parameter:
       time_step(int): the historial sequence length that we want to use for testing.
       Note: RNN use a sequence of data to predict.
       Returns: Numpy array of X (input) and y (output)'''
    x = []
    y = []

    for i in range(self.trainLength - time_step):
      datax, datay = self.getNextWindow(i, time_step)
      x.append(datax)
      y.append(datay)

    return np.array(x), np.array(y)

  def getTestSet(self, time_step):
    '''Create x and y test data sequences.
       Parameter:
       time_step(int): the historial sequence length that we want to use for testing.
       Note: RNN use a sequence of data to predict.
       Returns: Numpy array of X (input) and y (output) '''

    sequences = []
    for i in range(self.testLength - time_step):
      sequences.append(self.testData[i:i+time_step])

    sequences = np.array(sequences)
    sequences = self.normalise(sequences, False)  # Normalize the data

    x = []
    y = []
    for data in sequences:
      x.append(data[:-1])
      #go to the last sequence, get the first index because we want to use the close price to predict
      y.append(data[-1, [0]])
    return np.array(x), np.array(y)

  def getNextWindow(self, index, time_step):
    '''Normalise each window for the training data set.
       Parameters:
       index(int): the index of the input according to the data set.
       time_step(int): the historial sequence length that we want to use.
       Returns:
       x: The numpy array of sequences.
       y: The numpy array of last sequence.'''

    window = self.trainData[index:index + time_step]
    window = self.normalise(window, True)[0]
    x = window[:-1]
    y = window[-1, [0]]
    return x, y

  def normalise(self, window, singleWindow):
    '''Normalize each input by calculating the log return of the price.
       Parameters:
       window(): the array
       singleWindow(boolean): '''

    normalised = []
    if singleWindow == True:
      window = [window]
    for data in window:
      normalisedData = []
      for col in range(data.shape[1]):
        normalisedCol = []
        for row in data[:, col]:
          # getting the log return of stock price
          normalisedCol.append(np.log(float(row) / float(data[0, col])))
        normalisedData.append(normalisedCol)
      # convert it back to the original shape
      normalisedData = np.array(normalisedData).T
      normalised.append(normalisedData)
    return np.array(normalised)

  def getData(self, time_step):
    sequences = []
    for i in range(len(self.data) - time_step):
      sequences.append(self.data[i:i+time_step])

    sequences = np.array(sequences)
    sequences = self.normalise(sequences, False)  # Normalize the data

    x = []
    y = []
    for data in sequences:
      x.append(data[:-1])
      #go to the last sequence, get the first index because we want to use the close price to predict
      y.append(data[-1, [0]])
    return np.array(x), np.array(y)