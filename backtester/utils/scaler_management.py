from tkinter import filedialog
import customtkinter as ctk
import os

def load_scaler_file(path_container, indicator):
    """
    Opens a file dialog to select and load a scaler file (.pkl), updating the GUI to reflect the loaded scaler.

    Args:
    path_container (dict): Dictionary to store the path of the loaded scaler file.
    indicator (ctk.CTkLabel): GUI label that displays the loaded scaler file name.

    This function supports .pkl file formats for scalers, typically used in machine learning for data normalization or standardization.
    """
    
    scaler_file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if scaler_file_path:
        #print("Scaler loaded:", scaler_file_path)
        indicator.configure(text="Model Scaler " + os.path.basename(scaler_file_path) + " loaded")
        path_container['scaler_path'] = scaler_file_path 

def load_columns_file(path_container, indicator):
    """
    Opens a file dialog to select and load a scaler file (.pkl), updating the GUI to reflect the loaded scaler.

    Args:
    path_container (dict): Dictionary to store the path of the loaded scaler file.
    indicator (ctk.CTkLabel): GUI label that displays the loaded scaler file name.

    This function supports .pkl file formats for scalers, typically used in machine learning for data normalization or standardization.
    """
    
    columns_file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if columns_file_path:
        #print("Scaler loaded:", scaler_file_path)
        indicator.configure(text="Columns File " + os.path.basename(columns_file_path) + " loaded")
        path_container['columns_path'] = columns_file_path 

#loading scalar with prompt in process_data
def upload_scaler_prompt(value, frame, path_container, font):
    """
    Creates or updates the GUI elements for loading a scaler file based on user interaction.

    Args:
    value (str): The current value from a related GUI element that determines if the scaler UI should be shown.
    frame (ctk.CTkFrame): The frame where the scaler UI components are displayed.
    path_container (dict): Dictionary to store the path of the loaded scaler file.
    font (str): Font style for the button text.

    This function dynamically displays UI components for loading a scaler file if the user selects an option that requires uploading a scaler. It updates the visibility of the scaler loading button and indicator based on the user's selection.
    """

    scaler_indicator = ctk.CTkLabel(frame, text="No Scaler Model Selected")  # Moved outside to always ensure it's created

    def toggle_scaler_ui(show):
        if show:
            scaler_button.grid(row=0, column=2, padx=15, pady=15)
            scaler_indicator.grid(row=0, column=3, padx=15, pady=15)
        else:
            scaler_button.grid_forget()
            scaler_indicator.grid_forget()

    scaler_button = ctk.CTkButton(frame, text="Load Scaler (.pkl)", font=font, command=lambda: load_scaler_file(path_container, scaler_indicator))

    toggle_scaler_ui(value == "Upload Scaler")