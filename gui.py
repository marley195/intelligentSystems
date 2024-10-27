import tkinter as tk
from tkinter import messagebox
import numpy as np
import networkx as nx
from keras.models import load_model
from main import process_data, build_graph, find_routes, estimate_travel_time, eva_regress
from sklearn.preprocessing import MinMaxScaler

# Load models once for prediction
def load_models():
    custom_objects = {'MeanSquaredError': tf.keras.losses.MeanSquaredError, 'mse': tf.keras.losses.MeanSquaredError}
    models = {
        'LSTM': load_model('model/lstm.h5', custom_objects=custom_objects),
        'GRU': load_model('model/gru.h5', custom_objects=custom_objects),
        'SAEs': load_model('model/saes.h5', custom_objects=custom_objects),
        'RNN': load_model('model/simplernn.h5', custom_objects=custom_objects)
    }
    return models

models = load_models()

# SCATS Data and Graph Loading
def load_scats_data():
    data_file = '/Users/Ansh Sehgal/intelligentSystems/data/Scats Data October 2006.csv'
    lag = 12
    data, _, X_test, _, y_test, flow_scaler, _ = process_data(data_file, lag)
    return data, X_test, y_test, flow_scaler

data, X_test, y_test, flow_scaler = load_scats_data()
graph = build_graph(data, y_test.flatten())

def predict_volume(scats_number, model_name="LSTM"):
    model = models.get(model_name)
    if not model:
        return "Model not found"
    # Sample input for the prediction (adjust as needed)
    # Assuming X_test format; use real input if needed.
    X_sample = np.reshape(X_test[:1], (1, X_test.shape[1], 1))  # Adjust for model input
    prediction = model.predict(X_sample)
    prediction_rescaled = flow_scaler.inverse_transform(prediction)
    return f"Predicted volume for SCATS {scats_number} with {model_name}: {prediction_rescaled[0][0]}"

def gui_find_routes(origin, destination):
    origin = int(origin)
    destination = int(destination)
    routes = find_routes(graph, origin, destination)
    if not routes:
        return "No routes found."
    result = ""
    for i, route in enumerate(routes):
        travel_time = sum(graph[u][v]['weight'] for u, v in zip(route[:-1], route[1:]))
        result += f"Route {i + 1}: {route} - Estimated time: {travel_time:.2f} minutes\n"
    return result

# GUI Setup
def create_gui():
    root = tk.Tk()
    root.title("TFPS System")

    # Volume Prediction Section
    tk.Label(root, text="SCATS Volume Prediction", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, pady=10)

    tk.Label(root, text="SCATS Site Number:").grid(row=1, column=0, padx=10, pady=5)
    scats_entry = tk.Entry(root)
    scats_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Model:").grid(row=2, column=0, padx=10, pady=5)
    model_var = tk.StringVar(value="LSTM")
    model_options = ["LSTM", "GRU", "SAEs", "RNN"]
    tk.OptionMenu(root, model_var, *model_options).grid(row=2, column=1, padx=10, pady=5)

    def predict_action():
        scats_number = scats_entry.get()
        model_name = model_var.get()
        if scats_number:
            try:
                volume_prediction = predict_volume(int(scats_number), model_name=model_name)
                output_text.insert(tk.END, volume_prediction + "\n")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")
        else:
            messagebox.showwarning("Input Error", "Please enter a SCATS Site Number")

    tk.Button(root, text="Predict Volume", command=predict_action).grid(row=3, column=0, columnspan=2, pady=5)

    # Route Finder Section
    tk.Label(root, text="Route Finder", font=("Arial", 12)).grid(row=4, column=0, columnspan=2, pady=10)

    tk.Label(root, text="Origin SCATS Number:").grid(row=5, column=0, padx=10, pady=5)
    origin_entry = tk.Entry(root)
    origin_entry.grid(row=5, column=1, padx=10, pady=5)

    tk.Label(root, text="Destination SCATS Number:").grid(row=6, column=0, padx=10, pady=5)
    destination_entry = tk.Entry(root)
    destination_entry.grid(row=6, column=1, padx=10, pady=5)

    def route_action():
        origin = origin_entry.get()
        destination = destination_entry.get()
        if origin and destination:
            try:
                route_result = gui_find_routes(origin, destination)
                output_text.insert(tk.END, route_result + "\n")
            except Exception as e:
                messagebox.showerror("Error", f"Route finding failed: {e}")
        else:
            messagebox.showwarning("Input Error", "Please enter both Origin and Destination SCATS Numbers")

    tk.Button(root, text="Find Route", command=route_action).grid(row=7, column=0, columnspan=2, pady=5)

    # Output Display
    tk.Label(root, text="Output", font=("Arial", 12)).grid(row=8, column=0, columnspan=2, pady=10)
    output_text = tk.Text(root, height=10, width=50)
    output_text.grid(row=9, column=0, columnspan=2, padx=10, pady=5)

    root.mainloop()

# Run the GUI
create_gui()
