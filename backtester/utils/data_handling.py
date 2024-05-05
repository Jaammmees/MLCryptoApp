from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#used in multiple places --------------------
def load_data_file(path_container, indicator):
    """
    Opens a file dialog to select a data file and updates the GUI to reflect the loaded data.

    Args:
    path_container (dict): A dictionary to store the path of the loaded data file.
    indicator (ctk.CTkLabel): Label widget to show the loaded file name.

    The function supports CSV, Excel, and Parquet file formats.
    """

    data_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Parquet files", "*.parquet")])
    if data_file_path:
        #print("Data loaded:", data_file_path)
        indicator.configure(text="Data " + os.path.basename(data_file_path) + " loaded")
        path_container['data_path'] = data_file_path

#in data processing -------------------------
def load_data_file_and_modify(path_container, columns, data_indicator, columns_frame, column_list_frame, delete_column):
    """
    Loads data from a file and displays it in specified GUI frames with the option to delete columns.

    Args:
    path_container (dict): Container for file paths.
    columns (dict): Dictionary storing column widget references.
    data_indicator (ctk.CTkLabel): Label to indicate the status of data loading.
    columns_frame (ctk.CTkFrame): Frame to display column labels.
    column_list_frame (ctk.CTkFrame): Frame to manage column data and deletion buttons.
    delete_column (function): Function to call for deleting a column.

    This function also handles exceptions if the file loading fails.
    """

    data_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Parquet files", "*.parquet")])
    path_container['data_path'] = data_file_path
    if data_file_path:
        data_indicator.configure(text="Data " + os.path.basename(data_file_path) + " loaded")
        try:
            data = pd.read_csv(data_file_path) if data_file_path.endswith('.csv') \
                   else pd.read_excel(data_file_path) if data_file_path.endswith('.xlsx') \
                   else pd.read_parquet(data_file_path)

            for widget in columns_frame.winfo_children() + column_list_frame.winfo_children():
                widget.destroy()

            display_data_columns(data, columns, columns_frame, column_list_frame, delete_column)
        except Exception as e:
            data_indicator.configure(text=f"Failed to load data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

def display_data_columns(data, columns, columns_frame, column_list_frame, delete_column):
    """
    Displays data columns and initializes widgets for each column in the GUI.

    Args:
    data (DataFrame): Pandas DataFrame containing the data to display.
    columns (dict): Dictionary to store references to column widgets.
    columns_frame (ctk.CTkFrame): Frame to display column labels.
    column_list_frame (ctk.CTkFrame): Frame to display column widgets and deletion buttons.
    delete_column (function): Function to delete a column.

    This function creates labels and buttons for each column in the data.
    """
    
    for i, column in enumerate(data.columns):
        print(column)
        label = ctk.CTkLabel(columns_frame, text=column, width=20, padx=20)
        label.grid(row=0, column=i)
        frame = ctk.CTkFrame(column_list_frame, height=50)
        frame.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        label = ctk.CTkLabel(frame, text=column)
        label.pack(side="left", fill="x", expand=True, padx=10)
        delete_button = ctk.CTkButton(frame, text="X", width=40, height=40, command=lambda col=column: delete_column(col, columns))
        delete_button.pack(side="right")
        columns[column] = frame

        # Display a few rows of data
    num_rows_to_display = min(5, len(data))  # Show up to 5 rows
    for row_index in range(num_rows_to_display):
        for col_index, column in enumerate(data.columns):
            value = data.iloc[row_index, col_index]
            cell_label = ctk.CTkLabel(columns_frame, text=str(value), width=20)
            cell_label.grid(row=row_index + 1, column=col_index, padx=20)


#data preview after processing -------------------------
def display_data_preview(scaled_data, data_preview_frame):
    """
    Displays a preview of the processed data in the specified GUI frame.

    Args:
    scaled_data (DataFrame): Scaled data to be previewed.
    data_preview_frame (ctk.CTkFrame): Frame where the data preview is displayed.

    This function displays the first few rows (select num of rows) of the processed data.
    """
    # Clear previous data in the frame
    for widget in data_preview_frame.winfo_children():
        widget.destroy()

    # Displaying first few rows of data
    num_rows = min(10, len(scaled_data))  # Limit the number of rows to display
    header = ' | '.join(scaled_data.columns)
    header_label = ctk.CTkLabel(data_preview_frame, text=header, text_color="#FFFFFF",anchor='w')
    header_label.pack(side="top", fill='x', padx=10, pady=2)

    for row_index in range(num_rows):
        row_data = scaled_data.iloc[row_index]
        row_text = ' | '.join(str(x) for x in row_data)
        row_label = ctk.CTkLabel(data_preview_frame, text=row_text, text_color="#FFFFFF", anchor='w')
        row_label.pack(side="top", fill='x', padx=10, pady=2)

    data_preview_frame.pack(side="left", fill="both", expand=True)
    

#processign and saving the data -------------------------
def process_and_save_data(path_container, columns, data_preview_frame, choose_scaler_combo):
    """
    Processes the loaded data using a selected scaler and saves the scaled data.

    Args:
    path_container (dict): Dictionary containing paths to data and scaler files.
    columns (dict): Dictionary of active columns to process.
    data_preview_frame (ctk.CTkFrame): Frame to display processed data preview.
    choose_scaler_combo (ctk.CTkComboBox): Combo box to select the scaler type.

    Raises:
    Exception: If no data is loaded or no valid scaler is selected.

    This function scales the data and provides an option to save the processed data.
    """
    try:
        original_data_path = path_container['data_path']
        if not original_data_path:
            raise Exception("Data not loaded")
        
        data = pd.read_csv(original_data_path) if original_data_path.endswith('.csv') \
            else pd.read_excel(original_data_path) if original_data_path.endswith('.xlsx') \
            else pd.read_parquet(original_data_path)

        current_columns = list(columns.keys())
        data = data[current_columns]

        scaler_choice = choose_scaler_combo.get()
        if scaler_choice == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_choice == 'StandardScalar':
            scaler = StandardScaler()
        elif scaler_choice == 'Upload Scaler' and 'scaler_path' in path_container:
            with open(path_container['scaler_path'], 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise Exception("No valid scaler selected or loaded")
        
        scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        ask_fileName = ctk.CTkInputDialog(text = "Name for Processed Data", title = "New Data File")
        if ask_fileName:
            filepath = f'./processed_data/{ask_fileName.get_input()}.csv'
            scaled_data.to_csv(filepath, index = False)

        display_data_preview(scaled_data, data_preview_frame)  # Display the data in a simplified format

    except Exception as e:
        error_label = ctk.CTkLabel(data_preview_frame, text="Error: " + str(e), fg_color="#FFFFFF")
        error_label.pack(pady=15, padx=15, fill='both', expand=True)