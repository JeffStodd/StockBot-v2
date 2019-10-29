import pandas as pd
import tensorflow as tf
import numpy as np

'''
Same general structure as original stock bot
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

    model = tf.keras.models.load_model('Orig.h5') #load from presaved model
    #model = genModel()

    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam'
    )
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['binary_accuracy'])
    #train using batch size 64 and
    model.fit(inputs, outputs, batch_size = 64, epochs=0, verbose=2)

    #generate validation data input and output sets
    validationData = loadData("validate.csv")
    BTCData = loadData("BTC.csv")
    
    inputValidate = []
    outputValidate = []

    inputBTC = []
    outputBTC = []

    buildDataset(inputValidate, outputValidate, validationData)
    buildDataset(inputBTC, outputBTC, BTCData)

    return
    #testing on validation sets
    model.evaluate(inputValidate, outputValidate, verbose=2)
    model.evaluate(inputBTC, outputBTC, verbose=2)

    model.save('Orig.h5') #save trained model
    #model.save_weights('path_to_my_tf_checkpoint')

    #print(model.predict(training_data).round())

    #test through user input
    test(model, data, inputs, outputs)

#generate model
def genModel():
    model = tf.keras.Sequential() #feed forward
    #activation tahn to for -1 to 1 range
    model.add(tf.keras.layers.Dense(25, input_dim=7, activation='tanh')) #7 inputs
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(25, input_dim=25, activation='tanh')) #dense hidden layer with 25 nodes
    model.add(tf.keras.layers.Dense(1, input_dim=25, activation='tanh')) #output layer with 1 node

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
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
            outputs.append([1])
        elif future < 0: #else if bearish, set expected output to -1
            outputs.append([-1])
        else: #else the percent change is 0, set expected output to 0
            outputs.append([0])

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
    
    

main()
