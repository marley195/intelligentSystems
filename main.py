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


"""
    Notes:
    Current Model: LSTM is training really well, for some reason GRU/SAES are completely off on their predictions.
"""

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
        y_true: List/ndarray, true data.
        y_preds: List of ndarray, predicted data for each model.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=len(y_true), freq='15min')  # Make sure the length matches y_true

    fig, ax = plt.subplots()

    # Plot true data
    ax.plot(x, y_true, label='True Data')

    # Colors for each model (use colors without trailing spaces)
    colors = ['coral', 'violet', 'lightgreen']

    # Loop through each model's predictions and plot them
    for y_pred, name, color in zip(y_preds, names, colors):
        ax.plot(x, y_pred, label=name, color=color)

    # Formatting the plot
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    # Set time format for x-axis labels
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def main():
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    rnn = load_model('model/simplernn.h5')
    models = [lstm, gru, saes, rnn]
    names = ['LSTM', 'GRU', 'SAEs', 'RNN']
    periods = 288
    lag = 12
    data = '/Users/marleywetini/repos/intelligentSystems/data/Scats Data October 2006.csv'
    _, X_test, _, y_test, flow_scaler, _ = process_data(data, lag)
    y_preds = []
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
        y_pred_rescaled = flow_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = flow_scaler.inverse_transform(y_test.reshape(-1, 1))
        eva_regress(y_test_rescaled[:periods], y_pred_rescaled[:periods])
        print(name)
        y_preds.append(y_pred_rescaled[:periods].flatten())
        # Evaluate model performance
        print(name)
    # Plot results
    plot_results(y_test_rescaled[:periods], y_preds, names)



if __name__ == '__main__':
    main()