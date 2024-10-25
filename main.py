import math
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from geopy.distance import geodesic
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def MAPE(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    num = len(y_pred)
    sums = 0
    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp
    mape = sums * (100 / num)
    return mape

def eva_regress(y_true, y_pred):
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print(f'explained_variance_score: {vs}')
    print(f'mape: {mape}%')
    print(f'mae: {mae}')
    print(f'mse: {mse}')
    print(f'rmse: {math.sqrt(mse)}')
    print(f'r2: {r2}')

def plot_results(y_true, y_preds, names):
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=len(y_true), freq='15min')
    fig, ax = plt.subplots()
    ax.plot(x, y_true, label='True Data')
    colors = ['coral', 'violet', 'lightgreen']
    for y_pred, name, color in zip(y_preds, names, colors):
        ax.plot(x, y_pred, label=name, color=color)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.show()

def process_data(file_path, lag):
    data = pd.read_csv(file_path)
    print("Columns in the dataset:", data.columns)

    target_column = '00:15'
    y = data[target_column]

    if y.empty:
        raise ValueError(f"Target column '{target_column}' contains no valid data.")

    print("Target column sample data:\n", y.head())

    if 'SCATS Number' in data.columns:
        X = data.drop(columns=['SCATS Number', target_column, 'Location', 'Date'])
    else:
        raise KeyError("SCATS Number column not found.")

    flow_scaler = MinMaxScaler()
    y_scaled = flow_scaler.fit_transform(y.values.reshape(-1, 1))
    
    latlong_scaler = MinMaxScaler()
    if 'NB_LATITUDE' in data.columns and 'NB_LONGITUDE' in data.columns:
        latlong_scaled = latlong_scaler.fit_transform(data[['NB_LATITUDE', 'NB_LONGITUDE']])
    else:
        raise KeyError("Latitude and Longitude columns not found.")
    
    X_lagged = np.array([X.shift(lag).values for lag in range(1, lag + 1)])
    X_lagged = np.nan_to_num(X_lagged)
    X_train, X_test = X_lagged[:int(0.8 * len(X_lagged))], X_lagged[int(0.8 * len(X_lagged)):]
    y_train, y_test = y_scaled[:int(0.8 * len(y_scaled))], y_scaled[int(0.8 * len(y_scaled)):]
    
    return data, X_test, y_train, y_test, flow_scaler

def estimate_travel_time(vol_a, vol_b, dist, speed_limit=60, intersection_delay=30):
    avg_volume = (vol_a + vol_b) / 2
    base_time = dist / (speed_limit / 60)
    total_time = base_time + intersection_delay + (avg_volume / 100)
    return total_time

def build_graph(data, vol_data):
    G = nx.Graph()
    for i, row_a in data.iterrows():
        for j, row_b in data.iterrows():
            if i != j:
                dist = geodesic((row_a['NB_LATITUDE'], row_a['NB_LONGITUDE']),
                                (row_b['NB_LATITUDE'], row_b['NB_LONGITUDE'])).km
                if dist <= 1:
                    travel_time = estimate_travel_time(vol_data[i], vol_data[j], dist)
                    G.add_edge(row_a['SCATS Number'], row_b['SCATS Number'], weight=travel_time)
    return G

def find_routes(G, origin, destination):
    try:
        routes = list(nx.shortest_simple_paths(G, source=origin, target=destination, weight='weight'))[:5]
    except nx.NetworkXNoPath:
        routes = []
    return routes

def main():
    custom_objects = {'MeanSquaredError': tf.keras.losses.MeanSquaredError, 'mse': tf.keras.losses.MeanSquaredError}
    lstm = load_model('model/lstm.h5', custom_objects=custom_objects)
    gru = load_model('model/gru.h5', custom_objects=custom_objects)
    saes = load_model('model/saes.h5', custom_objects=custom_objects)
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']
    periods = 288
    lag = 12
    data_file = '/Users/Ansh Sehgal/intelligentSystems/data/Scats Data October 2006.csv'
    data, X_test, _, y_test, flow_scaler = process_data(data_file, lag)
    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test_res = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
        else:
            X_test_res = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_pred = model.predict(X_test_res)
        y_pred_rescaled = flow_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = flow_scaler.inverse_transform(y_test.reshape(-1, 1))
        eva_regress(y_test_rescaled[:periods], y_pred_rescaled[:periods])
        y_preds.append(y_pred_rescaled[:periods].flatten())
    plot_results(y_test_rescaled[:periods], y_preds, names)

    origin = int(input("Enter the origin SCATS number: "))
    destination = int(input("Enter the destination SCATS number: "))
    G = build_graph(data, y_test_rescaled.flatten())
    routes = find_routes(G, origin, destination)

    if not routes:
        print("No routes found between the specified SCATS numbers.")
        return

    for i, route in enumerate(routes):
        print(f"Route {i + 1}: {route}")
        total_time = sum(G[u][v]['weight'] for u, v in zip(route[:-1], route[1:]))
        print(f"Estimated travel time: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()
