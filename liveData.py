import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import plotly.graph_objects as go
import pandas_datareader as pd
import datetime
import tensorflow as tf
import numpy as np
import discord
import sys

'''
Discord Bot
'''
class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

        output = predict() #message to be sent
        for server in self.guilds:
            for channel in server.channels:
                if channel.name == 'general': #send into all general channels in all servers
                    await channel.send(output)
                    break
        sys.exit() #exit to free GPU memory, run bat file via task scheduler daily or run manually to predict again


'''
Returns string of prediction day, and bullish/bearish confidence values from live data
'''
def predict():    
    start = datetime.datetime.now()
    end = datetime.datetime.today() - datetime.timedelta(days=365)
    data = pd.DataReader('^SPX', 'stooq', start, end) #pulls data from SPX, start/end seem to be bugged
    #print(data[0:365])
    
    model = tf.keras.models.load_model('Model/Conv.h5') #load from presaved model

    dataArray = data.iloc[:,3][0:366] #access close day column for 366 days (to get 365 percent change days)

    '''
    convert to chronological order
    '''
    temp = []
    for x in reversed(dataArray):
        temp.append(x)

    '''
    convert to percentages
    '''
    inputArray = []
    for i in range(365):
        inputArray.append((temp[i+1]-temp[i])/temp[i])

    '''
    normalize data by local absolute max
    '''
    localMax = 0
    for i in range(365):
        localMax = max(localMax, abs(inputArray[i]))

    for x in inputArray:
        x = x/localMax

    '''
    format array for convolutional input
    '''
    inputArray = np.array([inputArray])
    inputArray = np.expand_dims(inputArray, axis = 2)
    
    output = model.predict(inputArray) #run prediction

    '''
    del unused for now
    '''
    del model
    tf.keras.backend.clear_session()
    
    del temp
    del inputArray
    del dataArray

    '''
    string to be returned
    '''
    strOut = "Closing Prediction for: " + (str)((data.index[0] + datetime.timedelta(days=1)).date())
    strOut = strOut + "\nBullish: " + '{percent:.2%}'.format(percent=output[0][0])
    strOut = strOut + "\nBearish: " + '{percent:.2%}'.format(percent=output[0][1])
    return strOut
    #print("Closing Prediction for:",(data.index[0] + datetime.timedelta(days=1)).date())
    #print("Bullish:", '{percent:.2%}'.format(percent=output[0][0]))
    #print("Bearish:", '{percent:.2%}'.format(percent=output[0][1]))

    '''
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

    fig.show()
    '''

def main():
    path = "D:\API Tokens\TradingBot.txt" #file containing bot api token, different for every developer
    with open(path) as f:
        token = f.readline()
    client = MyClient()
    client.run(token)
    
if __name__ == "__main__":
    main()
