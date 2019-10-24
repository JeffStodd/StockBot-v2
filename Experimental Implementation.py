import pandas as pd
import tensorflow as tf
import numpy as np
import os
import random

'''
Same inputs as original stock bot
2 output nodes for bearish and bullish confidence levels
'''

def loadData(path):
    data = pd.read_csv(path, names = ["Change"])
    return data

def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   
    data = loadData("Data.csv") #load data from csv
    
    inputs = [] #array or input arrays (size 7)
    outputs = [] #array of output arrays (size 1)
    
    buildDataset(inputs, outputs, data)

    #generate validation data input and output sets
    validationData = loadData("validate.csv")
    BTCData = loadData("BTC.csv")
    
    inputValidate = []
    outputValidate = []

    inputBTC = []
    outputBTC = []

    buildDataset(inputValidate, outputValidate, validationData)
    buildDataset(inputBTC, outputBTC, BTCData)

    model = tf.keras.models.load_model('Model.h5') #load from presaved model
    #model = genModel()

    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam'
    )
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['binary_accuracy', tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

    allInputs = inputBTC
    allOutputs = outputBTC
    for i in range(1688):
        orig = random.randint(0, len(inputs)-1)
        validate = random.randint(0, len(inputValidate)-1)
        allInputs.append(inputs[orig])
        allInputs.append(inputValidate[validate])
        allOutputs.append(outputs[orig])
        allOutputs.append(outputValidate[validate])

    sets = list(zip(allInputs, allOutputs))
    random.shuffle(sets)

    allInputs, allOutputs = zip(*sets)
    '''
    allInputs = inputs + inputValidate
    allInputs = inputs + inputBTC

    allOutputs = outputs + outputValidate
    allOutputs = outputs + outputBTC
    '''
    
    #train using batch size 64
    model.fit(allInputs, allOutputs, batch_size = 64, epochs=0, verbose=2)
    
    #testing on validation sets
    model.evaluate(inputs, outputs, verbose=2)
    model.evaluate(inputValidate, outputValidate, verbose=2)
    model.evaluate(inputBTC, outputBTC, verbose=2)

    model.save('Model.h5') #save trained model
    #model.save_weights('path_to_my_tf_checkpoint')

    
    #os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    #return
    
    #print(model.predict(training_data).round())

    #simulate(model, BTCData, inputBTC, outputBTC, 1, 0.75, 0, 0)
    #test through user input
    test(model, data, inputs, outputs)

#generate model
def genModel():
    model = tf.keras.Sequential() #feed forward
    #activation tahn to for -1 to 1 range
    model.add(tf.keras.layers.Dense(25, input_dim=7, activation='tanh')) #7 inputs
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(2, input_dim=25, activation='sigmoid')) #output layer with 1 node
    return model

def buildDataset(inputs, outputs, data):
    #build dataset
    #ignore last 7 days because can't extract an input/output value out of range
    for i in range(len(data["Change"]) - 7):
        temp = []
        m = -100 #local max
        for j in range(i, i+7): #getting max among local inputs
            m = max(m, abs(data["Change"][j]))
        m = max(m, abs(data["Change"][i+7])) #check if output is a max
        for j in range(i, i+7):
            temp.append(data["Change"][j]/m) #insert into temp input array
        inputs.append(temp) #insert input array into array of inputs
        future = data["Change"][i+7] #value we want to predict
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
            return
        predict(model, data, inputs, outputs, user)
    
#predict model on a single input array
def predict(model, data, inputs, outputs, day):
    print("\nTesting on:\n",data[day:day+7])
    output = model.predict([inputs[day]])
    print("Bullish/Bearish Confidence: ", output)
    print("Expected Confidence: ", outputs[day])
    print("Actual Percent Change: ", data["Change"][day+7])

def simulate(model, data, inputs, outputs, buyIn, threshold, entryDay, exitDay):
    money = buyIn
    asset = 0
    for i in range(entryDay, exitDay):
        prev = money + asset
        a = model.predict([inputs[i]])
        asset = asset + asset * float(data["Change"][i+6])
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
