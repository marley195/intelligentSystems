"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape. 

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

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
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

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
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
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

def main():
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    lag = 12
    data = '/Users/marleywetini/repos/intelligentSystems/data/Scats Data October 2006.csv'
    _, X_test, _, y_test, flow_scaler, latlong_scaler = process_data(data, lag)
    y_test = np.vectorize(flow_scaler.inverse_transform(y_test))

    for name, model in zip(names, models):
        # Reshape X_test based on the model requirements
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Plot the model structure
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        y_pred = model.predict(X_test)
        # Now, rescale the predictions back to their original values
        y_pred_rescaled = flow_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = flow_scaler.inverse_transform(y_test.reshape(-1, 1))
        #Rescale latitude and longitude data (if needed)
        X_test_latlong = X_test[:, :2]  # Extract the latlong columns (first two columns)
        X_test_latlong_rescaled = latlong_scaler.inverse_transform(X_test_latlong)



        """
        print(name)
        # Model prediction
        print(f'predicted shape: {predicted.shape}')
        print(f'y_test shape {y_test.shape}')
        # Check shape before inverse transform
        print(f'Predicted shape for {name}: {predicted.shape}')
        # Append first 288 predictions
        y_preds.append(predicted[:periods])

        # Evaluate model performance
        print(name)
        eva_regress(y_test, predicted)

    # Plot results
    plot_results(y_test[:periods], y_preds, names)
"""
if __name__ == '__main__':
    main()