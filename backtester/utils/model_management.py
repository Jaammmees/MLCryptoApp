from tkinter import filedialog
import pandas as pd
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

from utils.sequence_processing import create_sequences

def load_model_file(path_container, indicator):
    """
    Opens a file dialog to select and load a model file (.h5), updating the GUI to reflect the loaded model.

    Args:
    path_container (dict): Dictionary to store the path of the loaded model file.
    indicator (Widget): GUI widget (typically a label) to display the loaded model file name.

    This function supports .h5 file formats for models.
    """

    model_file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
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

    model_file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
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

    model_file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
    if model_file_path:
        model_indicator.configure(text="Model " + os.path.basename(model_file_path) + " loaded")
        path_container['model_path'] = model_file_path
        display_model_summary(model_file_path, model_preview_frame, font)

def start_training(path_container, sequence_preview_text, sequence_in, learning_rate, epochs):
    """
    Starts the training process for a loaded model with the specified dataset and hyperparameters.

    Args:
    path_container (dict): Dictionary storing paths to the model and data files.
    sequence_preview_text (Widget): Widget to display messages related to the training process.
    sequence_in (str): Entry widget content specifying the sequence length for training.
    learning_rate (float): Learning rate for the model optimizer.
    epochs (int): Number of training epochs.

    This function handles model training, including compiling and fitting the model, and updates the GUI with the training status.
    """

    if 'model_path' not in path_container or 'data_path' not in path_container:
        sequence_preview_text.configure(text="Model or data file not loaded.")
        return

    # Load model
    model = load_model(path_container['model_path'])

    # Fetch hyperparameters
    try:
        learning_rate = float(learning_rate)
        epochs = int(epochs)
    except ValueError:
        sequence_preview_text.configure(text="Invalid hyperparameters.")
        return

    # Load and prepare data
    data = pd.read_csv(path_container['data_path'])
    sequence_length = int(sequence_in)  # Ensure this entry is validated earlier in the workflow
    X, y = create_sequences(data, sequence_length, prediction_steps=5, target_column='close')

    # Split data into training and testing
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Compile the model
    #still to change such that we actually take the inputs we make
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mean_squared_error',
                metrics=['mean_absolute_error'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

    # Update GUI post-training
    sequence_preview_text.configure(text=f"Training complete. Final validation loss: {history.history['val_loss'][-1]}")

    # Optionally save the trained model
    model.save('./models/updated_model.h5')