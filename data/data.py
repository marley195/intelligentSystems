"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split

def process_flow_data(flow_data, max_len=None):
    """
    This function processes the flow data by flattening and padding/truncating to ensure uniform length.
    """
    # Flatten the lists inside each flow entry and find the maximum length if not provided
    flattened_flow_data = [np.array(x).flatten() for x in flow_data]
    if max_len is None:
        max_len = max(len(arr) for arr in flattened_flow_data)  # Find the max length of any array
    
    # Pad or truncate each array to the max length
    padded_flow_data = np.array([np.pad(arr, (0, max_len - len(arr)), 'constant') 
                                 if len(arr) < max_len else arr[:max_len] 
                                 for arr in flattened_flow_data])
    return padded_flow_data


def process_data(df_path, lags):
    # Step 1: Load and clean the data
    df = pd.read_csv(df_path, header=1).fillna(0)
    # Step 2: Define the flow columns (V00 to V95 for 96 time intervals)
    flow_columns = [f"V{str(i).zfill(2)}" for i in range(96)]
    grouped_data = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])[flow_columns].apply(lambda x: x.values.tolist())
    flow_data = grouped_data.values # shape=(140,0) | pandas
    flow_data = np.array(flow_data) # shape=(140,0) | numpy.ndarray
    #new_flow_data is used to allocate the Min/Max for data scaler
    new_flow_data = process_flow_data(flow_data)

    data_scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaler = data_scaler.fit(new_flow_data.reshape(-1, 1))
    latlong_scaler = MinMaxScaler(feature_range=(0, 1))
    latlong_data = grouped_data.reset_index()[['NB_LATITUDE', 'NB_LONGITUDE']]
    latlong_scaler = MinMaxScaler(feature_range=(0, 1)).fit(latlong_data)
    latlong_scaled = latlong_scaler.transform(latlong_data)
    train_data = []
    targets = []
    for i, flow in enumerate(flow_data):
        flow = np.array(flow).flatten()
        flow = data_scaler.transform(flow.reshape(-1,1)).reshape(1, -1)
        for j in range(0, len(flow[0]) - lags):  # Iterating over each possible lag of 12
            lagged_flow = flow[0][j:j+lags]  # Get the flow data for a lag of 12
            # Attach corresponding latlong data
            latlong = latlong_scaled[i]  # Get the lat/long for the current location //# to check latlong/flow data is correctly being added. 
            # Combine latlong and flow data
            combined_arr = np.hstack((latlong, lagged_flow))
            # Append the combined array to the training data
            train_data.append(combined_arr)

            # Optionally, define your target variable 'y' (e.g., the next traffic flow value)
            if j + lags < len(flow[0]):
                targets.append(flow[0][j + lags])  # Taking the next flow value as the target
    
    train_data = np.array(train_data)


    ##testing for data accruacy
    #first_two_cols = train_data[:, :2]
    #first_two_cols_scaled = latlong_scaler.inverse_transform(first_two_cols)
    #remaining_cols = train_data[:, 2:]
    #remaining_cols_scaled = data_scaler.inverse_transform(remaining_cols)
    #combined_data = np.hstack((first_two_cols_scaled, remaining_cols_scaled))
    #df_train_data = pd.DataFrame(combined_data)
    #df_train_data.to_csv("Scaled_data_review.csv")
    #print(combined_data)
    #
    X = np.array(train_data)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=.75)
    print(f"X train shape:{X_train.shape}")
    print(f" Y train shape{y_train.shape}")
    return X_train, X_test, y_train, y_test, data_scaler, latlong_scaler
