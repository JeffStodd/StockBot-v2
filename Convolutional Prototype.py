import pandas as pd
import tensorflow as tf
import numpy as np
import random
import threading
import matplotlib.pyplot as plt

'''
Inputs changed from weekly values (7 days) to year long (365 days)
Same outputs as previous models
Convolutional network implementation
Multithreading utilized for simulation (still slow)
'''

def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    epochCount = 0

    print("Loading Model")
    model = tf.keras.models.load_model('Model/Conv.h5') #load from presaved model
    #model = genModel() #uncomment to generate new model

    #loading raw data for simulation
    print("\nRaw Data")
    print("-----------------------------------------------------------------")
    recentData = loadDataRaw("Data/recent.csv") #unused
    data = loadDataRaw("Data/Data.csv") #S&P 2000-2019
    validationData = loadDataRaw("Data/Data2.csv") #S&P 1980-2000
    BTCData = loadDataRaw("Data/BTC.csv") #BTC data where volume > 0

    #loading different datasets
    print("\nDatasets")
    print("-----------------------------------------------------------------")
    validateRecent = loadData("Data/validateRecent.csv")
    train = loadData("Data/train.csv") 
    validate = loadData("Data/validate.csv")
    validateBTC = loadData("Data/validateBTC.csv")

    print("-----------------------------------------------------------------")
    print(model.summary()) #preview of model structure
    print("-----------------------------------------------------------------")
    
    #training settings
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam'
    )
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['binary_accuracy', tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

    '''
    Inputs and outputs to be trained on
    Add extra data here
    '''
    print("Inserting preformatted data")
    inputs = []
    outputs = []
    for i in range(len(train)):
        inputs.append(train.loc[i][0:365])
        outputs.append(train.loc[i][365:367])

    inputValidate = []
    outputValidate = []
    for i in range(len(validate)):
        inputValidate.append(validate.loc[i][0:365])
        outputValidate.append(validate.loc[i][365:367])

    inputBTC = []
    outputBTC = []
    for i in range(len(validateBTC)):
        inputBTC.append(validateBTC.loc[i][0:365])
        outputBTC.append(validateBTC.loc[i][365:367])

    recentInput = []
    recentOutput = []
    for i in range(len(validateRecent)):
        recentInput.append(validateRecent.loc[i][0:365])
        recentOutput.append(validateRecent.loc[i][365:367])
    

    allInputs = inputs + inputBTC
    allOutputs = outputs + outputBTC
                         
    '''
    End of generating training set
    '''

    '''
    Shuffle dataset and format for training method
    Expand dims
    Cast to np array
    '''
    print("Shuffling dataset")
    allInputs = np.array(allInputs)
    allOutputs = np.array(allOutputs)
    
    sets = list(zip(allInputs, allOutputs))
    random.shuffle(sets)

    allInputs, allOutputs = zip(*sets)

    '''
    Balance outputs
    '''
    print("Balancing data")
    trainIn = []
    trainOut = []

    numBullish = 0
    numBearish = 0
    for i in range(len(allOutputs)):
        if allOutputs[i][0] == 1:
            numBullish = numBullish + 1
        elif allOutputs[i][1] == 1:
            numBearish = numBearish + 1

    midway = min(numBullish, numBearish)
    
    numBullish = 0
    numBearish = 0
    for i in range(len(allOutputs)):
        if allOutputs[i][0] == 1:
            numBullish = numBullish + 1
            if numBullish <= midway:
                trainIn.append(allInputs[i])
                trainOut.append(allOutputs[i])
        elif allOutputs[i][1] == 1:
            numBearish = numBearish + 1
            if numBearish <= midway:
                trainIn.append(allInputs[i])
                trainOut.append(allOutputs[i])

    allInputs = np.array(trainIn)
    allOutputs = np.array(trainOut)
    
    allInputs = np.expand_dims(allInputs, axis=2)


    '''
    End of formatting and shuffling data
    '''
    print("Starting training")
    #train using batch size 64
    model.fit(allInputs, allOutputs, batch_size = 64, epochs=epochCount, use_multiprocessing = True, verbose=2)
    print("Saving model")
    model.save('Model/Conv.h5') #save trained model
    #testing on validation sets

    '''
    Formatting datasets to be tested
    '''
    inputs = np.expand_dims(inputs, axis=2)
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    inputValidate = np.expand_dims(inputValidate, axis=2)
    inputValidate = np.array(inputValidate)
    outputValidate = np.array(outputValidate)

    inputBTC = np.expand_dims(inputBTC, axis=2)
    inputBTC = np.array(inputBTC)
    outputBTC = np.array(outputBTC)

    recentInput = np.expand_dims(recentInput, axis=2)
    recentInput = np.array(recentInput)
    recentOutput = np.array(recentOutput)
    
    '''
    End of formatting datasets
    '''

    '''
    Evaluating model accuracy on different sets
    '''
    print("\n-----------------------------------------------------------------")
    print("Recent S&P Results")
    model.evaluate(recentInput, recentOutput, verbose=2)
    print("2000 - 2019 Results:")
    model.evaluate(inputs, outputs, verbose=2)
    print("1990 - 2000 Results:")
    model.evaluate(inputValidate, outputValidate, verbose=2)
    print("BTC Results:")
    model.evaluate(inputBTC, outputBTC, verbose=2)
    '''
    End of evaluating model
    '''
    
    #os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    #return
    
    #print(model.predict(training_data).round())

    '''
    Simulate automated trading using the model on given data
    '''
    print("Running simulation")

    #pass data to result arrays to be plotted
    results = []
    results2 = []
    results3 = []
    results4 = []
    
    threads = []
    
    t1 = threading.Thread(target=simulate, args=(results, model, data, inputs, outputs, 1, 0.75, 0, len(inputs)))
    t1.start()
    threads.append(t1)
    
    t2 = threading.Thread(target=simulate, args=(results2, model, validationData, inputValidate, outputValidate, 1, 0.75, 0, len(inputValidate)))
    t2.start()
    threads.append(t2)

    t3 = threading.Thread(target=simulate, args=(results3, model, BTCData, inputBTC, outputBTC, 1, 0.75, 0, len(inputBTC)))
    t3.start()
    threads.append(t3)

    t4 = threading.Thread(target=simulate, args=(results4, model, recentData, recentInput, recentOutput, 1, 0.75, 0, len(recentInput)))
    t4.start()
    threads.append(t4)

    #rejoining simulation threads before plotting
    for proc in threads:
        proc.join()

    plt.figure()
    plt.plot(results[0], results[1], label='Automated Trading')
    plt.plot(results[0], results[2], label='S&P 500 (2000-2019)')
    plt.legend(loc='upper left')

    plt.figure()
    plt.plot(results2[0], results2[1], label='Automated Trading')
    plt.plot(results2[0], results2[2], label='S&P 500 (1990-2000)')
    plt.legend(loc='upper left')

    plt.figure()
    plt.plot(results3[0], results3[1], label='Automated Trading')
    plt.plot(results3[0], results3[2], label='BTC')
    plt.legend(loc='upper left')

    plt.figure()
    plt.plot(results4[0], results4[1], label='Automated Trading')
    plt.plot(results4[0], results4[2], label='Recent S&P 500')
    plt.legend(loc='upper left')
    
    plt.show()
    '''
    End of simulating
    '''

    #test individual days via user input
    test(model, data, inputs, outputs)
    
#generate model
def genModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(365,1)))

    model.add(tf.keras.layers.Conv1D(kernel_size=3, filters=64, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2,strides=1))
    model.add(tf.keras.layers.Conv1D(kernel_size=3, filters=32, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2,strides=1))
    model.add(tf.keras.layers.Conv1D(kernel_size=3, filters=16, activation='tanh'))
    
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv1D(kernel_size=1, filters=8, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2,strides=1))
    
    model.add(tf.keras.layers.Conv1D(kernel_size=1, filters=4, activation='tanh'))
    model.add(tf.keras.layers.Conv1D(kernel_size=3, filters=8, activation='tanh'))
    model.add(tf.keras.layers.Conv1D(kernel_size=1, filters=4, activation='tanh'))

    model.add(tf.keras.layers.AveragePooling1D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv1D(kernel_size=1, filters=16, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2,strides=1))

    model.add(tf.keras.layers.Conv1D(kernel_size=1, filters=8, activation='tanh'))
    model.add(tf.keras.layers.Conv1D(kernel_size=3, filters=16, activation='tanh'))
    model.add(tf.keras.layers.Conv1D(kernel_size=1, filters=8, activation='tanh'))
    
    
    
    '''
    Tweak 2nd last layers and last layer
    Fix last layer stride = 1/2 and filters = 5/10
    
     
    model.add(tf.keras.layers.Conv1D(kernel_size=3, filters=5, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=3,strides=1))

    model.add(tf.keras.layers.Conv1D(kernel_size=3, filters=10, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=3,strides=2))

    model.add(tf.keras.layers.Conv1D(kernel_size=1, filters=10, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=3,strides=1))
    '''
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='sigmoid')) #output layer with 2 nodes
    return model

#loads data from csv file to rows of inputs + outputs
def loadData(path):
    print("Loading", path, end="...", flush=True)
    data = pd.read_csv(path)
    print(" Done")
    return data

#loads data from path into a csv file, column is named "Change"
def loadDataRaw(path):
    print("Loading", path, end="...", flush=True)
    data = pd.read_csv(path, names = ["Change"])
    print(" Done")
    return data

#test using user input
def test(model, data, inputs, outputs):
    user = 0
    print("Enter test day, -1 to exit")
    while True:
        user = int(input())
        if user < 0: #exit if negative
            break
        else:
            predict(model, data, inputs, outputs, user)
    
#predict model on a single input array
def predict(model, data, inputs, outputs, day):
    print("\nTesting on:\n",data[day:day+365])
    output = model.predict(np.array([inputs[day]]))
    print("Bullish/Bearish Confidence: ", output)
    print("Expected Confidence: ", outputs[day])
    print("Actual Percent Change: ", data["Change"][day+365])

#simulate automated trading using given model and input data
def simulate(results, model, data, inputs, outputs, buyIn, threshold, entryDay, exitDay):
    money = (buyIn+0)
    market = (buyIn + 0)
    curr = 0
    asset = 0

    bot = [] #contains simulated gain/loss from trading
    economy = [] #contains market values
    day = [] #contains i values
    for i in range(entryDay, exitDay):
        prev = money + asset #prev day (initialized to buyIn)
        prevMarket = market #prev day market price (initialized to buyIn)
        
        day.append(i)
        bot.append(prev)
        economy.append(prevMarket)
        
        a = model.predict(np.array([inputs[i]]))
        asset = asset + asset * float(data["Change"][i+365]) #modify asset based on price change if asset > 0
        market = market + market * float(data["Change"][i+365]) #modify market price based on price change
        if a[0][0] > threshold: #buy flag
            asset = asset + money
            money = 0
        elif a[0][1] > threshold: #sell flag
            money = money + asset
            asset = 0
        curr = money + asset #total value held by bot

        '''
        if curr < prev:
            print("Loss: ", curr, "\tMarket: ", market)
        else:
            print("Gain: ", curr, "\tMarket: ", market)
        '''
    print("Bot: ", curr, "\tMarket: ", market) #prints end values
    
    results.append(day)
    results.append(bot)
    results.append(economy)

#simulate automated trading using given model and input data
def simulateLowRisk(results, model, data, inputs, outputs, buyIn, threshold, entryDay, exitDay):
    money = (buyIn+0)
    market = (buyIn + 0)
    curr = 0
    asset = 0

    bot = [] #contains simulated gain/loss from trading
    economy = [] #contains market values
    day = [] #contains i values
    for i in range(entryDay, exitDay):
        prev = money + asset #prev day (initialized to buyIn)
        prevMarket = market #prev day market price (initialized to buyIn)
        
        day.append(i)
        bot.append(prev)
        economy.append(prevMarket)
        
        a = model.predict(np.array([inputs[i]]))
        asset = asset + asset * float(data["Change"][i+365]) #modify asset based on price change if asset > 0
        market = market + market * float(data["Change"][i+365]) #modify market price based on price change
        if a[0][0] > threshold: #buy flag
            asset = asset + money*(0.01)
            money = money - money*(0.01)
        elif a[0][1] > threshold: #sell flag
            money = money + asset*(0.5)
            asset = asset*(0.5)
        curr = money + asset #total value held by bot

        '''
        if curr < prev:
            print("Loss: ", curr, "\tMarket: ", market)
        else:
            print("Gain: ", curr, "\tMarket: ", market)
        '''
    print("Bot: ", curr, "\tMarket: ", market) #prints end values
    
    results.append(day)
    results.append(bot)
    results.append(economy)

if __name__ == '__main__':
    main()
