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
    """Mean Absolute Percentage Error"""
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
    """Evaluate the predicted result."""
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot the true data and predicted data."""
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


# Helper to build the graph for SCATS route guidance
def estimate_travel_time(vol_a, vol_b, dist, speed_limit=60, intersection_delay=30):
    avg_volume = (vol_a + vol_b) / 2
    base_time = dist / (speed_limit / 60)  # Time in minutes
    total_time = base_time + intersection_delay + (avg_volume / 100)  # Simplified approximation
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
                    G.add_edge(row_a['Unnamed: 1'], row_b['Unnamed: 1'], weight=travel_time)
    return G


def find_routes(G, origin, destination):
    try:
        routes = list(nx.shortest_simple_paths(G, source=origin, target=destination, weight='weight'))[:5]
    except nx.NetworkXNoPath:
        routes = []
    return routes


def process_data(file_path, lag):
    """Process SCATS data, extract features and labels, and apply scaling."""

    data = pd.read_csv(file_path)
    print("Columns in the dataset:", data.columns)

    # Drop any non-numeric columns and ensure data is numeric
    numeric_data = data.apply(pd.to_numeric, errors='coerce')

    # Remove rows with any NaN values that may have been introduced
    numeric_data = numeric_data.dropna()

    print("Processed numeric dataset columns:", numeric_data.columns)

    # Ensure the target column exists
    target_column = '22:15'
    if target_column not in numeric_data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset.")

    # Check the target column for missing or invalid values
    y = numeric_data[target_column]
    print(f"Checking target column '{target_column}'...")
    print(f"Number of valid entries: {y.count()}")
    print(f"First few values in target column: \n{y.head()}")

    if y.empty or y.count() == 0:
        raise ValueError(f"Target column '{target_column}' contains no valid data.")

    flow_scaler = MinMaxScaler()
    latlong_scaler = MinMaxScaler()

    # Ensure the SCATS number and coordinates exist and are valid
    if 'Unnamed: 1' not in numeric_data.columns or 'NB_LATITUDE' not in numeric_data.columns or 'NB_LONGITUDE' not in numeric_data.columns:
        raise KeyError("Required columns for SCATS Number or location not found.")

    X = numeric_data.drop(columns=['Unnamed: 1'])
    y_scaled = flow_scaler.fit_transform(y.values.reshape(-1, 1))
    latlong_scaled = latlong_scaler.fit_transform(numeric_data[['NB_LATITUDE', 'NB_LONGITUDE']])

    X_lagged = np.array([X.shift(lag).values for lag in range(1, lag+1)])
    X_lagged = np.nan_to_num(X_lagged)

    X_train, X_test = X_lagged[:int(0.8*len(X_lagged))], X_lagged[int(0.8*len(X_lagged)):]
    y_train, y_test = y_scaled[:int(0.8*len(y_scaled))], y_scaled[int(0.8*len(y_scaled)):]

    return numeric_data, X_test, y_train, y_test, flow_scaler, latlong_scaler


def main():
    # Load models with custom objects, replacing 'mse' with MeanSquaredError
    lstm = load_model('model/lstm.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError(), 'LSTM': tf.keras.layers.LSTM})
    gru = load_model('model/gru.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError(), 'GRU': tf.keras.layers.GRU})
    saes = load_model('model/saes.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError(), 'LSTM': tf.keras.layers.LSTM})
    
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    lag = 12
    data_file = 'C:/Users/Ansh Sehgal/intelligentSystems/data/Scats Data October 2006.csv'
    data, X_test, _, y_test, flow_scaler, latlong_scaler = process_data(data_file, lag)

    # Reshape and apply inverse transform
    y_test_rescaled = flow_scaler.inverse_transform(y_test.reshape(-1, 1))

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test_res = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test_res = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        y_pred = model.predict(X_test_res)
        y_pred_rescaled = flow_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_preds.append(y_pred_rescaled)

        # Evaluate model
        print(f"\n{name} Model Performance:")
        eva_regress(y_test_rescaled, y_pred_rescaled)

    # Plot results
    plot_results(y_test_rescaled[:288], [y_pred[:288] for y_pred in y_preds], names)

    # Graph-based Route Guidance
    origin = int(input("Enter the origin SCATS number: "))
    destination = int(input("Enter the destination SCATS number: "))

    G = build_graph(data, y_test_rescaled.flatten())
    routes = find_routes(G, origin, destination)

    if not routes:
        print("No routes found between the specified SCATS numbers.")
        return

    # Output routes and their travel times
    for i, route in enumerate(routes):
        print(f"Route {i + 1}: {route}")
        total_time = sum(G[u][v]['weight'] for u, v in zip(route[:-1], route[1:]))
        print(f"Estimated travel time: {total_time:.2f} minutes")


if __name__ == '__main__':
    main()
