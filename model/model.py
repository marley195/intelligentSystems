#Defination of NN model

from keras.layers import Dense, Dropout, Activation, LSTM, GRU, SimpleRNN
from keras.models import Sequential

def get_rnn(units):
    """simple RNN model


     # Arguments
        units: List(int), number of input, output, and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(SimpleRNN(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model

def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='Encoder'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output, name='Decoder', activation='sigmoid'))

    return model

def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output, name='Decoder', activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """

    sae1 = _get_sae(layers[0], layers[1], layers[0])  # Sae1: Input=12, Hidden=10, Output=12
    sae2 = _get_sae(layers[1], layers[2], layers[0])  # Sae2: Input=10, Hidden=8, Output=1
    sae3 = _get_sae(layers[2], layers[3], layers[0])  # Sae3: Input=8, Hidden=4, Output=1

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))  # Input=12, Hidden=10
    saes.add(Activation('relu'))
    saes.add(Dense(layers[2], name='hidden2'))  # Hidden=8
    saes.add(Activation('relu'))
    saes.add(Dense(layers[3], name='hidden3'))  # Hidden=4
    saes.add(Activation('relu'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))  # Output=1

    models = [sae1, sae2, sae3, saes]
    return models