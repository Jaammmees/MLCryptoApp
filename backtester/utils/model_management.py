from tkinter import filedialog
import pandas as pd
import os
from tensorflow.keras.models import load_model
from keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import customtkinter as ctk
import threading

from utils.sequence_processing import create_sequences

def load_model_file(path_container, indicator):
    """
    Opens a file dialog to select and load a model file (.h5), updating the GUI to reflect the loaded model.

    Args:
    path_container (dict): Dictionary to store the path of the loaded model file.
    indicator (Widget): GUI widget (typically a label) to display the loaded model file name.

    This function supports .h5 file formats for models.
    """

    model_file_path = filedialog.askopenfilename(initialdir="./models", filetypes=[("HDF5 files", "*.h5")])
    if model_file_path:
        #print("Model loaded:", model_file_path)
        indicator.configure(text="Model " + os.path.basename(model_file_path) + " loaded")
        path_container['model_path'] = model_file_path

def load_model_file_return_shapes(path_container, indicator, shape_indicator):
    """
    Loads a model from file and updates the GUI to show the shapes of the model's input and output.

    Args:
    path_container (dict): Dictionary to store the model path.
    indicator (Widget): Widget to display the name of the loaded model.
    shape_indicator (Widget): Widget to display the input and output shapes of the model.

    Returns:
    tuple: A tuple containing the input and output shapes of the model, or (None, None) if no model is loaded.

    This function also returns the model's input and output shapes for further use.
    """

    model_file_path = filedialog.askopenfilename(initialdir="./models", filetypes=[("HDF5 files", "*.h5")])
    if model_file_path:
        indicator.configure(text="Model " + os.path.basename(model_file_path) + " loaded")
        path_container['model_path'] = model_file_path
        model = load_model(path_container['model_path'])
        shape_indicator.configure(text=f"Shape of Model Input is: {model.input_shape}, \n and Model Output is: {model.output_shape}")
        shape_indicator.grid()
        return model.input_shape, model.output_shape  # Return the shapes
    return None, None

def load_model_preview(path_container, model_indicator, model_preview_frame, display_model_summary, font):
    """
    Loads a model file and displays its summary in the GUI.

    Args:
    path_container (dict): Dictionary storing the model path.
    model_indicator (Widget): Widget to display the model load status.
    model_preview_frame (Widget): Frame to display the model summary.
    display_model_summary (function): Function to generate and display the model summary.
    font (str): Font specification for the text display.

    This function updates the GUI with a preview of the model's structure.
    """

    model_file_path = filedialog.askopenfilename(initialdir="./models", filetypes=[("HDF5 files", "*.h5")])
    if model_file_path:
        model_indicator.configure(text="Model " + os.path.basename(model_file_path) + " loaded")
        path_container['model_path'] = model_file_path
        display_model_summary(model_file_path, model_preview_frame, font)

optimisers = {
    'adam': Adam,
    'sgd': SGD,
    'rmsprop': RMSprop
    #add more if needed
}

class GUIProgress(Callback):
    def __init__(self, root, progress_bar, progress_bar_label):
        self.root = root
        self.progress_bar = progress_bar
        self.progress_bar_label = progress_bar_label

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the current progress
        current_progress = (epoch + 1) / self.params['epochs'] * 100
        self.progress_bar_label.configure(text = f"Epoch {epoch + 1}/{self.params['epochs']}")
        # Schedule the update to be run in the main thread
        self.root.after(0, self.update_progress_bar, current_progress)

    def update_progress_bar(self, value):
        # Convert the percentage to a value between 0 and 1
        scaled_value = value / 100
        self.progress_bar.set(scaled_value)  # Set the progress bar value
        self.progress_bar.update_idletasks()


def start_training(path_container, validation_loss_label, sequence_in, sequence_out, optimiser, loss_type, metrics_list, learning_rate, epochs, saved_label, root, progress_bar, progress_bar_label):
    """
    Starts the training process for a loaded model with the specified dataset and hyperparameters.

    Args:
    path_container (dict): Dictionary storing paths to the model and data files.
    validation_loss_label (Widget): Widget to display messages related to the training process.
    sequence_in (str): Entry widget content specifying the sequence length for training.
    learning_rate (float): Learning rate for the model optimizer.
    epochs (int): Number of training epochs.

    This function handles model training, including compiling and fitting the model, and updates the GUI with the training status.
    """

    

    if 'model_path' not in path_container or 'data_path' not in path_container:
        validation_loss_label.configure(text="Model or data file not loaded.")
        return

    # Load model
    model = load_model(path_container['model_path'])

    # Fetch hyperparameters
    try:
        learning_rate = float(learning_rate)
        epochs = int(epochs)
    except ValueError:
        validation_loss_label.configure(text="Invalid hyperparameters.")
        return

    # Load data
    data = pd.read_csv(path_container['data_path'])

    # Define sequence parameters
    sequence_length = int(sequence_in)
    sequence_out = int(sequence_out)

    print(sequence_out)
    print("Taking in", sequence_length, "units of data with", sequence_out, "features to predict the next 1 unit of close price")

    # Extract column names excluding the target column
    include_columns = [col for col in data.columns if col != 'Close' and col != 'open_time']
    # Create sequences
    X, y = create_sequences(data, sequence_length, sequence_out, target_column='Close', include_columns=include_columns)

    print(X.shape)  # Expected shape: (num_samples, sequence_length, num_features)
    print(y.shape)  # Expected shape: (num_samples,)

    # Split data into training and testing
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Get the optimizer from the dictionary
    optimiser_class = optimisers.get(optimiser.lower(), Adam)
    optimiser = optimiser_class(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimiser, loss=loss_type, metrics=[metrics_list])

    # Progress Bar
    gui_progress = GUIProgress(root, progress_bar, progress_bar_label)
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1, callbacks=[gui_progress])

    metrics_summary = ""
    # Get the last epoch's results
    last_epoch_metrics = {key: values[-1] for key, values in history.history.items()}
    # Create a formatted string of metric results
    for metric, value in last_epoch_metrics.items():
        metrics_summary += f"{metric}: {value:.4f}, "
    # Remove the last comma and space
    metrics_summary = metrics_summary.rstrip(', ')

    # Update GUI post-training
    validation_loss_label.configure(text=f"Training complete. Metrics: {metrics_summary}")

    ask_fileName = ctk.CTkInputDialog(text = "Name for Trained Model", title = "New Trained Model")
    if ask_fileName:
        model.save(f'./models/{ask_fileName.get_input()}.h5')
        progress_bar.grid_forget()
        progress_bar_label.grid_forget()
        saved_label.configure(text=f"Model Successfully Saved")

    
    # Optionally save the trained model
    