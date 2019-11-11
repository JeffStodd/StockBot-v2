import csv
import pandas as pd
import threading

def main():

    #loading different datasets
    recent = loadData("recent.csv") #unused
    data = loadData("Data.csv") #S&P 2000-2019
    validationData = loadData("Data2.csv") #S&P 1980-2000
    BTCData = loadData("BTC.csv") #BTC data where volume > 0

    print("Building Set")

    train = []
    
    validate = []

    validateBTC = []

    threads = []
    
    t = threading.Thread(target=buildDataset, args=(train, data))
    t.start()
    threads.append(t)
    
    t = threading.Thread(target=buildDataset, args=(validate, validationData))
    t.start()
    threads.append(t)

    t = threading.Thread(target=buildDataset, args=(validateBTC, BTCData))
    t.start()
    threads.append(t)

    #rejoin dataset threads before writing
    for proc in threads:
        proc.join()
    
    with open('train.csv', 'w', newline='') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerows(train)

    with open('validate.csv', 'w', newline='') as f: 
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerows(validate)

    with open('validateBTC.csv', 'w', newline='') as f:
        wtr = csv.writer(f, delimiter= ',')
        wtr.writerows(validateBTC)


#builds a dataset given data is greater than 365
#multithreading utilized to speed up the for loop
def buildDataset(rows, data):
    #build dataset
    #ignore last 7 days because can't extract an input/output value out of range
    threads = []
    for i in range(len(data["Change"]) - 365):
        t = threading.Thread(target=buildDatasetHelper, args=(rows, data, i))
        t.start()
        threads.append(t)
        
    for proc in threads:
        proc.join()
        
    print("Done building set")

#used in buildDataset
def buildDatasetHelper(rows, data, i):
        temp = []
        m = -100 #local max
        for j in range(i, i+365): #getting max among local inputs
            m = max(m, abs(data["Change"][j]))
        m = max(m, abs(data["Change"][i+365])) #check if output is a max
        for j in range(i, i+365):
            temp.append(data["Change"][j]/m) #insert into temp input array
        future = data["Change"][i+365] #value we want to predict
        if future > 0: #if bullish, set expected output to 1
            temp.append(1)
            temp.append(0)
        elif future < 0: #else if bearish, set expected output to -1
            temp.append(0)
            temp.append(1)
        else: #else the percent change is 0, set expected output to 0
            temp.append(0)
            temp.append(0)
        rows.append(temp)

#loads data from csv, column is named "Change"
def loadData(path):
    print("Loading", path, end="...", flush=True)
    data = pd.read_csv(path, names = ["Change"])
    print(" Done")
    return data

if __name__ == '__main__':
    main()
