import numpy as np
import pandas as pd

def create_sequences(data, sequence_length, prediction_steps, target_column='Close', include_columns=None):
    """
    Generates sequences from time series data suitable for training sequence prediction models.

    Args:
    data (DataFrame): The dataset containing the time series data.
    sequence_length (int): The number of time steps per sequence.
    prediction_steps (int): The number of steps ahead to predict.
    target_column (str, optional): The column from `data` to predict. Defaults to 'Close'.
    include_columns (list, optional): List of column names to include in the input data. If None, only `target_column` is used.

    Returns:
    tuple: A tuple containing two numpy arrays, X and y. X is an array of input sequences, and y is the corresponding target values at `prediction_steps` into the future.

    This function processes the data to create overlapping sequences (windows) of length `sequence_length`, and corresponding targets that are `prediction_steps` away from the end of each sequence.
    """

    X, y = [], []
    data = data.dropna().reset_index(drop=True)  # Reset index after dropping NA values
    
    # Select columns if specified, otherwise default to using only the target column
    if include_columns is not None:
        data = data[include_columns + [target_column]]
    else:
        data = data[[target_column]]
        
    for i in range(sequence_length, len(data) - prediction_steps + 1):
        X.append(data.iloc[i-sequence_length:i].values)  # Adjust based on actual columns to include
        y.append(data[target_column].pct_change(prediction_steps).iloc[i + prediction_steps - 1])
        
    return np.array(X), np.array(y)

def generate_sequence(input_shape, sequence_in, sequence_indicator, path_container):
    """
    Generates and displays a preview of sequences based on the input data and model's expected input shape.

    Args:
    input_shape (tuple): The shape that the model expects for its input data.
    sequence_in (str): The length of the sequences to be generated, as entered by the user.
    sequence_indicator (Widget): A GUI widget used to display messages about the sequence generation process.
    path_container (dict): A dictionary storing paths to data files.

    This function validates the user-entered sequence length, checks if the data file is loaded, and if successful, generates and previews the data sequences.
    """
    # Validate sequence length entry
    if not sequence_in.isdigit():
        sequence_indicator.configure(text="Invalid sequence length.")
        return
    if 'data_path' not in path_container:
        sequence_indicator.configure(text="No data file loaded.")
        return
    print(input_shape)
    sequence_length = int(sequence_in)
    if sequence_length != input_shape[1]:  # assuming input_shape is like (None, sequence_length, num_features)
        sequence_indicator.configure(text=f"Sequence length mismatch. Model expects {input_shape[1]}.")
        return
    
    # Load data and handle possible exceptions
    try:
        data = pd.read_csv(path_container['data_path'])  # Load data, adjust based on actual data type
        target_column = 'close'  # Example: use a dropdown or text entry to set this in the GUI
        include_columns = None  # Example: this could be set via a multi-select list or checkboxes in the GUI

        # Generate sequences using the refactored function
        # Prediction steps is how many units into the future we are trying to predict
        X, y = create_sequences(data, sequence_length, prediction_steps=5, target_column=target_column, include_columns=include_columns)

        #print("Sample of X:", X[:1]) 
        #print("Sample of y:", y[:1])  

        # Update GUI to indicate successful sequence generation
        sequence_indicator.configure(text="Sequence generated successfully. Number of sequences: " + str(len(X)))
    except Exception as e:
        sequence_indicator.configure(text=f"Error: {str(e)}")
