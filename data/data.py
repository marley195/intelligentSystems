"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split

#def create_lagged_features(df, flow_columns, lags):
#    """Create lagged features for the specified flow columns."""
#    for lag in range(1, lags + 1):
#        lagged_cols = df[flow_columns].shift(lag)
#        lagged_cols.columns = [f"{col}_lag{lag}" for col in flow_columns]
#        df = pd.concat([df, lagged_cols], axis=1)
#    
#    # Drop the rows with missing values caused by lagging
#    df = df.dropna().reset_index(drop=True)
#    return df

def process_data(df_path, lags):
    # Step 1: Load and clean the data
    df = pd.read_csv(df_path, header=1).fillna(0)
    # Step 2: Define the flow columns (V00 to V95 for 96 time intervals)
    # Step 2: Define the flow columns (V00 to V95 for 96 time intervals)
    flow_columns = [f"V{str(i).zfill(2)}" for i in range(96)]
    grouped_data = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])[flow_columns].apply(lambda x: x.values.tolist())
    flow_data = grouped_data.values
    df_flow_data = np.array(flow_data)
    data_scaler = MinMaxScaler()
    latlong_scaler = MinMaxScaler()
    latlong_data = df[['NB_LATITUDE', 'NB_LONGITUDE']]
    latlong_scaled = latlong_scaler.fit_transform(latlong_data)
    train_data = []
    targets = []
    for i, flow in enumerate(df_flow_data):
        flow = np.array(flow).flatten()
        flow = data_scaler.fit_transform(flow.reshape(-1,1)).reshape(1, -1)
        for j in range(0, len(flow[0]) - lags):  # Iterating over each possible lag of 12
            lagged_flow = flow[0][j:j+lags]  # Get the flow data for a lag of 12
            # Attach corresponding latlong data
            latlong = latlong_scaled[i]  # Get the lat/long for the current location
            # Combine latlong and flow data
            combined_arr = np.hstack((latlong, lagged_flow))
            # Append the combined array to the training data
            train_data.append(combined_arr)

            # Optionally, define your target variable 'y' (e.g., the next traffic flow value)
            if j + lags < len(flow[0]):
                targets.append(flow[0][j + lags])  # Taking the next flow value as the target

    train_data = np.array(train_data)
    print(train_data.shape)

    X = np.array(train_data)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)

    return X_train, X_test, y_train, y_test, data_scaler, latlong_scaler






"""
    #Retrieve data from dataset and put into
    df_raw = pd.read_csv(data, encoding='utf-8', header=1).fillna(0)
    #drop columns which are not required.
    df_raw = df_raw.drop(['CD_MELWAY', 'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY', 'Unnamed: 106', 'Unnamed: 107', 'Unnamed: 108'], axis=1)

    df1_train, df2_test = train_test_split(df_raw, test_size=0.2, shuffle=True)

    #Create scaler and reshape dataframes for only values V00 - V95
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1_train.loc[:,'V00':'V95'].values.reshape(-1, 1))

    #Tranforming and reshaping Traffic flow data
    #    1 .reshap(-1, 1) - transforms the data values into a column vector | produces a 2D array
    #            - -1 infers the number of rows meaning the function can determine how many rows it needs
    #            - 1 means to create 1 column
    #    2 .reshap(-1, 1)[0] - reshapes the array into a row Vector | produces a 1D array
    #        - the (-1, 1) Does the same as above.
    #        - [0] flattens the array back into a 1D array by selecting the first row
    #1: produces a 
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
"""