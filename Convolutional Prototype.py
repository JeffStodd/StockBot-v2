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
Multithreading utilized for building datasets (still slow)
'''

def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    model = tf.keras.models.load_model('Conv.h5') #load from presaved model
    model = genModel() #uncomment to generate new model

    #loading different datasets
    recent = loadData("recent.csv") #unused
    data = loadData("Data.csv") 
    validationData = loadData("validate.csv")
    BTCData = loadData("BTC.csv")
    
    
    
    
    '''
    Generate 3 different datasets for training/testing/validation
    data = S&P last 20 years
    validationData = S&P 1990 - 2000
    BTC Data = recent BTC data (~3-4 years)

    Multithreading utilized to generate them synchronously 
    '''
    print("Building Set")

    inputs = [] #array or input arrays
    outputs = [] #array of output arrays
    
    inputValidate = []
    outputValidate = []

    inputBTC = []
    outputBTC = []

    threads = []
    
    t = threading.Thread(target=buildDataset, args=(inputs, outputs, data))
    t.start()
    threads.append(t)
    
    t = threading.Thread(target=buildDataset, args=(inputValidate, outputValidate, validationData))
    t.start()
    threads.append(t)

    t = threading.Thread(target=buildDataset, args=(inputBTC, outputBTC, BTCData))
    t.start()
    threads.append(t)

    '''
    End of generating datasets
    '''
    
    print(model.summary()) #preview of model structure

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

    #rejoin dataset threads before training
    for proc in threads:
        proc.join()


    '''
    Inputs and outputs to be trained on
    Add extra data here
    '''
    allInputs = inputValidate

    allOutputs = outputValidate
    '''
    End of generating training set
    '''

    

    '''
    Shuffle dataset and format for training method
    Expand dims
    Cast to np array
    '''
    print("Shuffling dataset")
    
    sets = list(zip(allInputs, allOutputs))
    random.shuffle(sets)

    allInputs, allOutputs = zip(*sets)
    allInputs = np.expand_dims(allInputs, axis=2)

    allInputs = np.array(allInputs)
    allOutputs = np.array(allOutputs)

    '''
    End of formatting and shuffling data
    '''
    
    #train using batch size 64
    model.fit(allInputs, allOutputs, batch_size = 64, epochs=1000, use_multiprocessing = True, verbose=2)
    model.save('Conv.h5') #save trained model
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
    '''
    End of formatting datasets
    '''

    '''
    Evaluating model accuracy on different sets
    '''
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
    
    t = threading.Thread(target=simulate, args=(results, model, data, inputs, outputs, 1, 0.75, 0, len(inputs)))
    t.start()
    threads.append(t)

    t = threading.Thread(target=simulate, args=(results2, model, validationData, inputValidate, outputValidate, 1, 0.75, 0, len(inputValidate)))
    t.start()
    threads.append(t)

    t = threading.Thread(target=simulate, args=(results3, model, BTCData, inputBTC, outputBTC, 1, 0.75, 0, len(inputBTC)))
    t.start()
    threads.append(t)

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
    
    model.add(tf.keras.layers.Conv1D(kernel_size=5, filters=10, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=5,strides=2))

    model.add(tf.keras.layers.Conv1D(kernel_size=5, filters=10, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=5,strides=2))

    model.add(tf.keras.layers.Conv1D(kernel_size=5, filters=10, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=5,strides=2))

    model.add(tf.keras.layers.Conv1D(kernel_size=5, filters=10, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=5,strides=2))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='sigmoid')) #output layer with 1 node
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
    return model

#loads data from path into a csv file, column is named "Change"
def loadData(path):
    print("Loading", path, end="...", flush=True)
    data = pd.read_csv(path, names = ["Change"])
    print(" Done")
    return data

#builds a dataset given data is greater than 365
#multithreading utilized to speed up the for loop
def buildDataset(inputs, outputs, data):
    #build dataset
    #ignore last 7 days because can't extract an input/output value out of range
    threads = []
    for i in range(len(data["Change"]) - 365):
        t = threading.Thread(target=buildDatasetHelper, args=(inputs, outputs, data, i))
        t.start()
        threads.append(t)
        
    for proc in threads:
        proc.join()
        
    print("Done building set")

#used in buildDataset
def buildDatasetHelper(inputs, outputs, data, i):
        temp = []
        m = -100 #local max
        for j in range(i, i+365): #getting max among local inputs
            m = max(m, abs(data["Change"][j]))
        m = max(m, abs(data["Change"][i+365])) #check if output is a max
        for j in range(i, i+365):
            temp.append(data["Change"][j]/m) #insert into temp input array
        inputs.append(temp) #insert input array into array of inputs
        future = data["Change"][i+365] #value we want to predict
        if future > 0: #if bullish, set expected output to 1
            outputs.append([1,0])
        elif future < 0: #else if bearish, set expected output to -1
            outputs.append([0,1])
        else: #else the percent change is 0, set expected output to 0
            outputs.append([0,0])
            
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
    money = market = buyIn
    curr = asset = 0

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

if __name__ == '__main__':
    main()
