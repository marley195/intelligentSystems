"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def process_data(data, lags):
    """Process data
    Reshape and split train est data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
 """
    #Retrieve data from dataset and put into
    df_raw = pd.read_csv(data, encoding='utf-8', header=1).fillna(0)
    #drop columns which are not required.
    df_raw = df_raw.drop(['CD_MELWAY', 'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY', 'Unnamed: 106', 'Unnamed: 107', 'Unnamed: 108'], axis=1)

    df1_train, df2_test = train_test_split(df_raw, test_size=1/3, shuffle=True)

    #Create scaler and reshape dataframes for only values V00 - V95
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1_train.loc[:,'V00':'V95'].values.reshape(-1, 1))
    flow1 = scaler.transform(df1_train.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2_test.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]



    #divided traffic data into Test/Train
    train, test = [], []  

    for i in range(lags, len(flow1)):   
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)   

    X_train = train[:, :-1]  # All columns except the last one
    y_train = train[:, -1]   # The last column
    X_test = test[:, :-1]    # All columns except the last one
    y_test = test[:, -1]     # The last column


    return X_train, y_train, X_test, y_test, scaler