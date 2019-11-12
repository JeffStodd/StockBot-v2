BTC.csv	- raw BTC percent change data historical (starts first day after volume > 0) until mid-late October
Data.csv - raw S&P 500 data from ~1990-2000
Data2.csv	- raw S&P 500 data from ~2000-2019 (mid-late October 2019)
recent.csv - raw S&P 500 data about 2 years back until mid-late October 2019

train.csv	- compiled Data2.csv into columns of 367 length rows of input days and output expectation (training data)
validate.csv	- compiled Data.csv into columns of 367 length rows of input days and output expectation (validation data)
validateBTC.csv	- compiled BTC.csv into columns of 367 length rows of input days and output expectation (training data)
validateRecent.csv - compiled recent.csv into columns of 367 length rows of input days and output expectation (validation data)
