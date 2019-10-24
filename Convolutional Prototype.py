import pandas as pd
import tensorflow as tf
import numpy as np
import random
import threading

'''
Inputs changed from weekly values (7 days) to year long (365 days)
Same outputs as previous models
Convolutional network implementation
Multithreading utilized for building datasets (still slow)
'''

def loadData(path):
    print("Loading Data")
    data = pd.read_csv(path, names = ["Change"])
    print("Done")
    return data

def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    model = tf.keras.models.load_model('Conv.h5') #load from presaved model
    #model = genModel()
    
    data = loadData("Data.csv") #load data from csv
    validationData = loadData("validate.csv")
    BTCData = loadData("BTC.csv")
    
    inputs = [] #array or input arrays (size 7)
    outputs = [] #array of output arrays (size 1)

    threads = []

    print("Building Set")

    
    t = threading.Thread(target=buildDataset, args=(inputs, outputs, data))
    t.start()
    threads.append(t)
    
    
    #buildDataset(inputs, outputs, data)

    #generate validation data input and output 
    
    inputValidate = []
    outputValidate = []

    inputBTC = []
    outputBTC = []

    
    t = threading.Thread(target=buildDataset, args=(inputValidate, outputValidate, validationData))
    t.start()
    threads.append(t)
    

    t = threading.Thread(target=buildDataset, args=(inputBTC, outputBTC, BTCData))
    t.start()
    threads.append(t)

    #buildDataset(inputValidate, outputValidate, validationData)
    #buildDataset(inputBTC, outputBTC, BTCData)

    
    print(model.summary())
    
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam'
    )
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['binary_accuracy', tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

    for proc in threads:
        proc.join()

    print("Combining subsets")
    allInputs = (inputBTC)
    allOutputs = (outputBTC)
    for i in range(len(inputBTC)):
        orig = random.randint(0, len(inputs)-1)
        validate = random.randint(0, len(inputValidate)-1)
        allInputs.append(inputs[orig])
        allInputs.append(inputValidate[validate])
        allOutputs.append(outputs[orig])
        allOutputs.append(outputValidate[validate])

    

    print("Shuffling dataset")

    sets = list(zip(allInputs, allOutputs))
    random.shuffle(sets)

    allInputs, allOutputs = zip(*sets)
    allInputs = np.expand_dims(allInputs, axis=2)

    allInputs = np.array(allInputs)
    allOutputs = np.array(allOutputs)
    
    '''
    allInputs = inputs + inputValidate
    allInputs = inputs + inputBTC

    allOutputs = outputs + outputValidate
    allOutputs = outputs + outputBTC
    '''
    
    #train using batch size 64
    model.fit(allInputs, allOutputs, batch_size = 64, epochs=0, verbose=2)
    model.save('Conv.h5') #save trained model
    #testing on validation sets
    
    inputs = np.expand_dims(inputs, axis=2)
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    
    inputValidate = np.expand_dims(inputValidate, axis=2)
    inputValidate = np.array(inputValidate)
    outputValidate = np.array(outputValidate)

    inputBTC = np.expand_dims(inputBTC, axis=2)
    inputBTC = np.array(inputBTC)
    outputBTC = np.array(outputBTC)

    model.evaluate(inputs, outputs, verbose=2)
    model.evaluate(inputValidate, outputValidate, verbose=2)
    model.evaluate(inputBTC, outputBTC, verbose=2)
    
    #os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    #return
    
    #print(model.predict(training_data).round())
    
    #test through user input
    print("Running simulation")
    #simulate(model, BTCData, inputBTC, outputBTC, 1, 0.75, 0, 1000)
    #simulate(model, data, inputs, outputs, 1, 0.75, 0, 7800)
    simulate(model, validationData, inputValidate, outputValidate, 1, 0.75, 0, 7800)
    
    test(model, data, inputs, outputs)
    
#generate model
def genModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(365,1)))
    
    model.add(tf.keras.layers.Conv1D(kernel_size=2, filters=128, activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(2,2))

    model.add(tf.keras.layers.Conv1D(kernel_size=2, filters=64, activation='relu'))
    model.add(tf.keras.layers.AveragePooling1D(2,2))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='sigmoid')) #output layer with 1 node
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
    return model

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
        if user < 0:
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

def simulate(model, data, inputs, outputs, buyIn, threshold, entryDay, exitDay):
    money = buyIn
    asset = 0
    for i in range(entryDay, exitDay):
        prev = money + asset
        a = model.predict(np.array([inputs[i]]))
        asset = asset + asset * float(data["Change"][i+365])
        if a[0][0] > threshold:
            asset = asset + money
            money = 0
        elif a[0][1] > threshold:
            money = money + asset
            asset = 0
        curr = money + asset
        if curr < prev:
            print("Loss: ", curr)
        else:
            print("Gain: ", curr)
    

main()
