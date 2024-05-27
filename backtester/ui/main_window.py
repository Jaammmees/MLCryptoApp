#library imports
import customtkinter as ctk
from tkinter import *
from tkcalendar import DateEntry
from PIL import Image
from keras.models import load_model
from utils.display_model_summary import display_model_summary
from utils.data_handling import load_data_file, load_data_file_and_modify, process_and_save_data, load_data_file_and_preview
from utils.model_management import load_model_file, load_model_preview, start_training, update_layer_param, update_layer_type, update_layer_widgets, build_and_save_model
from utils.scaler_management import load_scaler_file, upload_scaler_prompt, load_columns_file
from trading.historical import start_backtest
from trading.real_time import fetch_data, fetch_current_minute_data, plot_data, extend_future_data
import pandas as pd
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import joblib
import numpy as np
import matplotlib.pyplot as plt

import threading

class MainWindow(ctk.CTk):
    """
    Main application window for a machine learning backtester GUI built using customtkinter.

    The window includes a sidebar for navigation and a main frame where different functionalities are displayed,
    such as historical data loading, real-time data processing, model building, and training.

    Methods:
    create_widgets: Sets up widgets in the main frame (not implemented here).
    setup_sidebar: Initializes and configures sidebar with navigation buttons.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the main application window and setup the UI components.
        """
        super().__init__(*args, **kwargs)

        self.title('Machine Learning Backtester')
        self.geometry('1600x900')
        
        #initialise sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        #customise it using .pack
        self.sidebar.pack(side="left", fill="y", expand=False)

        #initialise mainframe
        self.main_frame = ctk.CTkScrollableFrame(self, corner_radius=0, fg_color=("#EDEDED", "#4B4B4B"))
        #customise
        self.main_frame.pack(side="right", expand=True, fill="both")
        
        #fonts
        self.navbar_font = ctk.CTkFont(family = "Helvetica", size = 12, weight = "bold")
        self.title_font = ctk.CTkFont(family = "Helvetica", size = 40, weight = "bold")
        self.button_font = ctk.CTkFont(family = "Helvetica", size = 15, weight = "bold")
        self.combo_box_font = ctk.CTkFont(family = "Helvetica", size = 20, weight = "bold")

        #specific to real-time trading and fetching, is a bool statement to stop or proceed.
        self.is_running = False

        self.setup_sidebar()

        #makes historical default
        self.load_historical()

    def setup_sidebar(self):
        """
        Sets up the sidebar with navigation buttons each linked to a specific functionality within the application.

        This method loads images for buttons, creates button widgets, and configures their grid placement in the sidebar.
        """

        #Images for navigation buttons
        chart_image = Image.open("./images/line-chart-svgrepo-com.png")
        dollar_image = Image.open("./images/dollar-sign-svgrepo-com.png")
        train_image = Image.open("./images/settings-svgrepo-com.png")
        data_image = Image.open("./images/data-svgrepo-com.png")
        build_image = Image.open("./images/build-svgrepo-com.png")
        self.backtestImage = ctk.CTkImage(dark_image = chart_image, size=(50,50))
        self.realTimeImage = ctk.CTkImage(dark_image = dollar_image, size=(50,50))
        self.trainImage = ctk.CTkImage(dark_image = train_image, size=(50,50))
        self.dataImage = ctk.CTkImage(dark_image = data_image, size=(50,50))
        self.buildImage = ctk.CTkImage(dark_image = build_image, size=(50,50))
        
        #Navigation Buttons
        button_backtest = ctk.CTkButton(self.sidebar, text="Backtest", font=self.navbar_font, compound=BOTTOM, command=self.load_historical, image=self.backtestImage, width=60, height=60)
        button_realtime = ctk.CTkButton(self.sidebar, text="Real-Time", font=self.navbar_font, compound=BOTTOM, command=self.load_realtime, image=self.realTimeImage, width=60, height=60)
        button_build = ctk.CTkButton(self.sidebar, text="Build Model", font=self.navbar_font, compound=BOTTOM, command=self.load_build_model, image=self.buildImage, width=60, height=60)
        button_data = ctk.CTkButton(self.sidebar, text="Process Data", font=self.navbar_font, compound=BOTTOM, command=self.load_process_data, image=self.dataImage, width=60, height=60)
        button_training = ctk.CTkButton(self.sidebar, text="Train Model", font=self.navbar_font, compound=BOTTOM, command=self.load_train_model, image=self.trainImage, width=60, height=60)

        #Navigation Button positioning
        button_backtest.grid(row=1, column=0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_realtime.grid(row=2, column=0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_build.grid(row=3, column=0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_data.grid(row=4, column=0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_training.grid(row=5, column=0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")

        #sidebar weighting to make it centered with 5 buttons
        self.sidebar.grid_columnconfigure(0, weight=1)  # Make column 0 take up all available space
        self.sidebar.grid_rowconfigure(0, weight=1)  # Spacer row at the top
        self.sidebar.grid_rowconfigure(1, weight=0)  # Actual button row
        self.sidebar.grid_rowconfigure(2, weight=0)  # Actual button row
        self.sidebar.grid_rowconfigure(3, weight=0)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(4, weight=0)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(5, weight=0)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(6, weight=1)  # Spacer row at the bottom

    #------------------------------------------------------------------------------------------------------------------------------------------------

    def load_historical(self):
        """
        Prepares the frame for historical data loading and backtesting operations within the application.

        This method handles the user interface setup for loading models, scalers, and historical data files,
        providing a GUI for initiating backtests. It includes creating and positioning buttons for loading
        necessary components and executing the backtest.

        Steps Performed:
        1. Clear existing widgets: This ensures the main frame is ready for new content.
        2. Display title: A label for "Backtesting" is set up to indicate the current operation.
        3. Setup loaders: Buttons and indicators for loading the machine learning model, scaler, and data file are created.
        4. Prepare for backtesting: Configures a frame with a button that should trigger the backtesting process (functionality to be linked).
        5. Result display setup: Initializes a frame intended to display the results of the backtest.

        Important GUI Elements Created:
        - backtest_title (ctk.CTkLabel): Label indicating the backtesting section.
        - loader_frame (ctk.CTkFrame): Contains buttons for loading the model, scaler, and data.
        - model_button, scaler_button, data_button (ctk.CTkButton): Buttons to trigger loading respective files.
        - model_indicator, scaler_indicator, data_indicator (ctk.CTkLabel): Labels to show load status.
        - backtest_button_frame (ctk.CTkFrame): Frame for the backtest initiation button.
        - backtest_button (ctk.CTkButton): Button to start the backtesting process (requires command linking).
        - backtest_result_frame (ctk.CTkFrame): Frame where backtest results would be displayed.
        """

        self.is_running = False  # stop the realtime process (very hacky i know)

        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        path_container = {}

        #title
        backtest_title = ctk.CTkLabel(self.main_frame, text="Backtesting", font=self.title_font, text_color="#353535")
        backtest_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        #loading frame
        loader_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        loader_frame.pack(pady=15, side=TOP, fill = X, padx = 20)

        #need to be able to load a machine model, and a scalar, choose the data we want to backtest it on LOCALLY or just a TICKER that we can grab
        #the data has to be in the form INDEX == Timestep, Open, High, Low, Close, Volume as of now
        #buttons and text for loading model, scaler, and data
        model_button = ctk.CTkButton(loader_frame, text="Load Model (.keras)", font=self.button_font, command=lambda : load_model_file(path_container, model_indicator))
        model_indicator = ctk.CTkLabel(loader_frame, text="No Model Selected")
        scaler_button = ctk.CTkButton(loader_frame, text="Load Scaler (.pkl)", font=self.button_font, command=lambda : load_scaler_file(path_container, scaler_indicator))
        scaler_indicator = ctk.CTkLabel(loader_frame, text="No Scaler Model Selected")
        columns_button = ctk.CTkButton(loader_frame, text="Load Columns (.pkl)", font=self.button_font, command=lambda : load_columns_file(path_container, columns_indicator))
        columns_indicator = ctk.CTkLabel(loader_frame, text="No Columns File Selected")
        data_button = ctk.CTkButton(loader_frame, text="Load Data (Excel/Parquet)", font=self.button_font, command=lambda : load_data_file(path_container, data_indicator))
        data_indicator = ctk.CTkLabel(loader_frame, text="No Data Selected")

        #button positioning
        model_button.grid(row=0, column = 0, padx=15, pady=15)
        model_indicator.grid(row = 1, column = 0, padx=15, pady=15)
        scaler_button.grid(row=0, column = 1, padx=15, pady=15)
        scaler_indicator.grid(row = 1, column = 1, padx=15, pady=15)
        columns_button.grid(row=0, column = 2, padx=15, pady=15)
        columns_indicator.grid(row = 1, column = 2, padx=15, pady=15)
        data_button.grid(row=0, column = 3, padx=15, pady=15)
        data_indicator.grid(row = 1, column = 3, padx=15, pady=15)
        
        #backtesting options
        backtest_options_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        backtest_options_frame.pack(pady=15,side=TOP,fill=X,padx=20)
        starting_balance_label = ctk.CTkLabel(backtest_options_frame, text="Starting Balance", font=self.button_font)
        starting_balance_label.grid(row=0,column=0, pady=15,padx=15)
        starting_balance_input = ctk.CTkEntry(backtest_options_frame, corner_radius=10)
        starting_balance_input.grid(row=0,column=1,padx=15,pady=15)

        backtest_button_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        backtest_button_frame.pack(pady=15, side=TOP, fill=X, padx=20)

        # Button to start backtesting
        backtest_button = ctk.CTkButton(
            backtest_button_frame, text="Run Backtest", font=self.button_font, 
            command=lambda: start_backtest(self, path_container, backtest_progressBar, backtest_progressBar_label, predictions_label, self.button_font, backtest_result_frame, starting_balance_input.get())
        )
        backtest_button.grid(row=0, column=0, padx=15, pady=15)

        # Progress bar for backtesting
        backtest_progressBar = ctk.CTkProgressBar(backtest_button_frame, orientation=HORIZONTAL, progress_color="#33cc33")
        backtest_progressBar.grid(row=0, column=1, padx=15, pady=15)
        backtest_progressBar.set(0)

        # Label for progress bar
        backtest_progressBar_label = ctk.CTkLabel(backtest_button_frame, text="", font=self.button_font)
        backtest_progressBar_label.grid(row=0, column=2, padx=15, pady=15)

        # Label for displaying predictions
        predictions_label = ctk.CTkLabel(backtest_button_frame, text="", font=self.button_font)
        predictions_label.grid(row=1, column=0, columnspan=3, padx=15, pady=15)

        #backtest results frame
        backtest_result_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        backtest_result_frame.pack(pady=10, side=TOP, fill = BOTH, padx = 20)

    #------------------------------------------------------------------------------------------------------------------------------------------------

    def load_build_model(self):
        """
        Loads the interface for building machine learning models (only LSTM atm) within the application.

        This method sets up the GUI components necessary for configuring and building models, including:
        - Selecting the model type.
        - Naming the model.
        - Configuring the number of layers.
        - Specifying parameters for each layer.
        - Setting hyperparameters.
        - Initiating the model build process and saving the model.

        Steps:
        1. Clear existing widgets: Prepares the main frame for new content.
        2. Display the title for the model building section.
        3. Configure the frame and dropdown for model type selection.
        4. Add entry fields for the model name.
        5. Set up a slider for selecting the number of layers.
        6. Create a frame for layer configuration and dynamically update it based on the slider value.
        7. Configure the frame for hyperparameter inputs such as sequence length and number of features.
        8. Add a button to initiate the model building process.
        9. Set up a frame for displaying the model summary after building.
        """
        self.is_running = False  # stop the realtime process

        # Wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Title
        build_model_title = ctk.CTkLabel(self.main_frame, text="Build Model", font=self.title_font, text_color="#353535")
        build_model_title.pack(pady=20, padx=25, side=TOP, anchor="w")

        # Select model frame
        select_model_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        select_model_frame.pack(pady=15, side=TOP, fill=X, padx=20)

        # Model Type Selection: Dropdown to choose between different types of models
        select_model_combo = ctk.CTkComboBox(select_model_frame, values=["LSTM", "CNN", "RNN"], height=50, width=150, font=self.combo_box_font)
        select_model_combo.grid(row=0, column=0, padx=15, pady=15, ipadx=30)

        # Model Name
        model_name_label = ctk.CTkLabel(select_model_frame, text="Model Name")
        model_name_label.grid(row=0, column=1, padx=15, pady=15)
        model_name_entry = ctk.CTkEntry(select_model_frame)
        model_name_entry.grid(row=0, column=2, padx=15, pady=15)

        # Slider for number of layers
        select_layers_text = ctk.CTkLabel(select_model_frame, text="Model Layers")
        select_layers_text.grid(row=0, column=3, padx=15, pady=15)
        select_layers = ctk.CTkSlider(select_model_frame, from_=1, to=10, number_of_steps=10, command=lambda value: self.slider_value(value, select_layers_value, self.building_layers_frame))
        select_layers.set(1)
        select_layers.grid(row=0, column=4, padx=15, pady=15, ipadx=30)
        select_layers_value = ctk.CTkLabel(select_model_frame, text="1")
        select_layers_value.grid(row=0, column=5, padx=15, pady=15)

        # Building layers frame
        self.building_layers_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.building_layers_frame.pack(pady=15, side=TOP, fill=X, padx=20)

        self.layer_widgets = []
        update_layer_widgets(self.building_layers_frame, self.layer_widgets, update_layer_type, update_layer_param, 1)

        # Hyperparameters and model preview
        parameters_title = ctk.CTkLabel(self.main_frame, text="Hyperparameter Configuration & Model Preview", font=self.combo_box_font, text_color="#353535")
        parameters_title.pack(padx=15, pady=15, side=TOP, anchor="w")

        side_by_side_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        side_by_side_frame.pack(pady=20, side=TOP, fill=X, padx=20)

        # Hyperparameters frame
        select_parameters_frame = ctk.CTkFrame(side_by_side_frame, corner_radius=10, fg_color='transparent', bg_color='transparent')
        select_parameters_frame.pack(pady=15, side=LEFT, fill=X, padx=20)

        # Sequence Length In
        sequence_length_in_label = ctk.CTkLabel(select_parameters_frame, text="Sequence Length In:")
        sequence_length_in_label.grid(row=1, column=0, padx=15, pady=15)
        sequence_length_in_entry = ctk.CTkEntry(select_parameters_frame)
        sequence_length_in_entry.grid(row=1, column=1, padx=15, pady=15)

        # Sequence Length Out
        sequence_length_out_label = ctk.CTkLabel(select_parameters_frame, text="Sequence Length Out:")
        sequence_length_out_label.grid(row=2, column=0, padx=15, pady=15)
        sequence_length_out_entry = ctk.CTkEntry(select_parameters_frame)
        sequence_length_out_entry.grid(row=2, column=1, padx=15, pady=15)

        # Number of Features
        number_of_features_label = ctk.CTkLabel(select_parameters_frame, text="Number of Features:")
        number_of_features_label.grid(row=3, column=0, padx=15, pady=15)
        number_of_features_entry = ctk.CTkEntry(select_parameters_frame)
        number_of_features_entry.grid(row=3, column=1, padx=15, pady=15)

        # Model preview frame
        model_preview_frame = ctk.CTkFrame(side_by_side_frame, corner_radius=10, fg_color='#333333')
        model_preview_frame.pack(pady=20, side=RIGHT, fill=X, padx=20)
        model_preview_frame.pack_forget()

        # Build model button
        model_build_button_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        model_build_button_frame.pack(pady=20, side=TOP, fill=X, padx=20)
        model_build_button = ctk.CTkButton(model_build_button_frame, text="Build Model", font=self.button_font, command=lambda: build_and_save_model(self.layer_widgets, sequence_length_in_entry, model_name_entry, number_of_features_entry, model_preview_frame, self.button_font))
        model_build_button.grid(row=0, column=0, padx=15, pady=15)



    def slider_value(self, value, select_layers_value, building_layers_frame):
        """
        Update the number of layers based on the slider value.

        Args:
            value (int): The number of layers selected.
            select_layers_value (CTkLabel): Label to display the selected number of layers.
            building_layers_frame (CTkFrame): Frame to hold the layer widgets.
        """
        int_value = int(value)
        select_layers_value.configure(text=f"{int_value}")
        update_layer_widgets(building_layers_frame, self.layer_widgets, update_layer_type, update_layer_param, int_value)


    #------------------------------------------------------------------------------------------------------------------------------------------------

    def load_process_data(self):
        """
        Sets up the GUI for data processing including model loading, data loading, and data preprocessing operations.

        This method handles the setup for:
        - Loading and displaying machine learning models.
        - Loading, modifying, and displaying data files.
        - Data normalization and scaling selection.
        - Initiating data processing and saving operations.

        Steps:
        1. Clear existing widgets: Ensures the main frame is ready for new widgets.
        2. Create and display the title for the data processing section.
        3. Configure frames and buttons for loading models and data, and for displaying model summaries.
        4. Allow dynamic loading and modification of data columns.
        5. Provide options for data normalization and scaling.
        6. Prepare and display the interface for initiating the data processing.
        """

        self.is_running = False  # stop the realtime process (very hacky i know)

        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        #internal functions
        self.columns = {}
        path_container = {}

        def delete_column(column, columns):
            # print(column, columns)
            print(f"trying to delete column {column}")
            if column in columns:
                columns[column].destroy()  # Remove the widget
                del columns[column]  # Remove the reference

        process_data_title = ctk.CTkLabel(self.main_frame, text="Process Data", font=self.title_font, text_color="#353535")
        process_data_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        #Data Upload:
        load_files_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        load_files_frame.pack(pady=15, side='top', fill='x', padx=20)
        load_files_frame.grid_columnconfigure(0, weight=1)  # Set equal weight if needed
        load_files_frame.grid_columnconfigure(1, weight=1)  # Set equal weight if needed

        #datepicker
        choose_date = ctk.CTkFrame(self.main_frame, corner_radius=10)
        choose_date.pack(pady=15,side=TOP,fill=X,padx=20)
        
        start_date_label = ctk.CTkLabel(choose_date, text="Start Date:")
        start_date_label.pack(side=LEFT, padx=15, pady=15)
        start_date_picker = DateEntry(choose_date, width=12, background='darkblue', foreground='white', borderwidth=2)
        start_date_picker.pack(side=LEFT, padx=15, pady=15)
        
        end_date_label = ctk.CTkLabel(choose_date, text="End Date:")
        end_date_label.pack(side=LEFT, padx=15, pady=15)
        end_date_picker = DateEntry(choose_date, width=12, background='darkblue', foreground='white', borderwidth=2)
        end_date_picker.pack(side=LEFT, padx=15, pady=15)

        # Button to load a model
        model_button = ctk.CTkButton(load_files_frame, text="Load Model (.keras)", font=self.button_font, command= lambda : load_model_preview(path_container, model_indicator, model_preview_frame, display_model_summary, self.button_font))
        model_button.grid(row=0, column=0, padx=15, pady=15, ipadx=20)  # Fill the cell

        # Indicator label for model loading
        model_indicator = ctk.CTkLabel(load_files_frame, text="No Model Selected")
        model_indicator.grid(row=1, column=0, padx=15, pady=15, sticky='nsew')  # Fill the cell

        # Outer frame for model preview (scrollable)
        model_preview_frame_outer = ctk.CTkFrame(load_files_frame, corner_radius=10)
        model_preview_frame_outer.grid(row=2, column=0, padx=15, pady=15, sticky='nsew')  # Fill the cell

        # Scrollable frame for model preview
        model_preview_frame = ctk.CTkScrollableFrame(model_preview_frame_outer, corner_radius=10, fg_color="#353535", orientation=VERTICAL)
        model_preview_frame.pack(padx=15, pady=15, expand=True, fill='both')  # Fill the outer frame

        #data
        data_button = ctk.CTkButton(load_files_frame, text="Load Data (Excel/Parquet)",
                                font=self.button_font, 
                                command=lambda: load_data_file_and_modify(path_container, self.columns,data_indicator, columns_frame, self.column_list_frame, delete_column, start_date_picker, end_date_picker))
        data_indicator = ctk.CTkLabel(load_files_frame, text="No Data Selected")
        data_button.grid(row=0, column = 1, padx=15, pady=15)
        data_indicator.grid(row = 1, column = 1, padx=15, pady=15)

        columns_frame = ctk.CTkScrollableFrame(load_files_frame, orientation=HORIZONTAL)
        columns_frame.grid(row=2, column=1, padx=15, pady=15, sticky="ew")

        #feature selection,
        data_configuration = ctk.CTkFrame(self.main_frame, corner_radius=10)
        data_configuration.pack(pady=15,side=TOP,fill=X,padx=20)

        #column modification
        modify_columns_text = ctk.CTkLabel(data_configuration, text = "Current Columns", font=self.button_font)
        modify_columns_text.pack(padx=15, side='left')
        self.column_list_frame = ctk.CTkScrollableFrame(data_configuration, corner_radius=10, orientation=HORIZONTAL, height=50)
        self.column_list_frame.pack(padx=15, pady=15, fill=X)

        # Data Scaling/Normalization:
        normalisation_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        normalisation_frame.pack(pady=15,side=TOP,fill=X,padx=20)
        # Combobox to choose between scaling methods (e.g., Min-Max Scaling, Standard Scaling).
        choose_scaler_text = ctk.CTkLabel(normalisation_frame, text="Choose Scaler")
        choose_scaler_text.grid(row=0,column=0,padx=15,pady=15)
        # provide option to upload own scaler file too, which will make it pop up the upload scaler button,
        choose_scaler_combo = ctk.CTkComboBox(normalisation_frame, values=['MinMaxScaler', 'StandardScaler', 'Upload Scaler', 'No Scaler'], 
                                      command=lambda value: upload_scaler_prompt(value, normalisation_frame, path_container, self.button_font))
        choose_scaler_combo.grid(row=0,column=1,padx=15,pady=15)

        # Sequence Preparation:
        # intake the model shape for the sequence length, crucial for LSTM models.
        data_process_button_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        data_process_button_frame.pack(pady=15,side=TOP,fill=X,padx=20)

        data_process_button = ctk.CTkButton(data_process_button_frame, font=self.button_font, text="Process Data & Save", command=lambda: process_and_save_data(path_container, self.columns, data_preview_frame, choose_scaler_combo, start_date_picker, end_date_picker))
        data_process_button.grid(row=0,column=0,padx=15,pady=15)

        data_preview_frame_outer = ctk.CTkFrame(self.main_frame,corner_radius=10)
        data_preview_frame_outer.pack(pady=15,padx=20, side=TOP,fill=X)

        processed_data_indicator = ctk.CTkLabel(data_preview_frame_outer, text=f"Preview of processed data", font=self.button_font)
        processed_data_indicator.pack(pady=15,padx=15,side=TOP,fill=X)

        data_preview_frame = ctk.CTkScrollableFrame(data_preview_frame_outer,corner_radius=10, fg_color="#353535")
        data_preview_frame.pack(pady=15,padx=15,side=TOP,fill=X)

    #------------------------------------------------------------------------------------------------------------------------------------------------

    def load_train_model(self):
        """
        Configures the main frame for model training operations, including loading data and models, setting up sequence generation,
        and configuring training hyperparameters.

        This method orchestrates the setup for:
        - Loading machine learning models and the corresponding data for training.
        - Specifying sequence generation parameters crucial for time-series model training.
        - Allowing the user to configure and initiate the training process.

        Steps:
        1. Clear the existing GUI components in the main frame to make space for new components.
        2. Create labels, entries, and buttons that allow the user to:
            - Load a model and visualize its input and output shapes.
            - Load the dataset for training.
            - Configure and apply sequence settings for the model.
            - Set training hyperparameters (like learning rate and epochs).
            - Initiate the training process.
        3. Display dynamic feedback on the model and data status
        """

        self.is_running = False  # stop the realtime process (very hacky i know)

        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        process_data_title = ctk.CTkLabel(self.main_frame, text="Model Training", font=self.title_font, text_color="#353535")
        process_data_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        path_container = {}
        self.model_input_shape = None
        self.model_output_shape = None

        def update_shapes():
            model = load_model(path_container['model_path'])
            print("Update_shapes being called" + path_container['model_path'])
            if model:
                print("yes")
                self.model_input_shape = model.input_shape
                self.model_output_shape = model.output_shape
                print(self.model_input_shape, self.model_output_shape)

        #Data Upload:
        load_files_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        load_files_frame.pack(pady=15, side='top', fill='x', padx=20)
        load_files_frame.grid_columnconfigure(0, weight=1)  # Set equal weight if needed
        load_files_frame.grid_columnconfigure(1, weight=1)  # Set equal weight if needed

        # Button to load a model
        model_button = ctk.CTkButton(load_files_frame, text="Load Model (.keras)", font=self.button_font, command= lambda : (load_model_preview(path_container, model_indicator, model_preview_frame, display_model_summary, self.button_font), update_shapes()))
        model_button.grid(row=0, column=0, padx=15, pady=15, ipadx=20)  # Fill the cell

        # Indicator label for model loading
        model_indicator = ctk.CTkLabel(load_files_frame, text="No Model Selected")
        model_indicator.grid(row=1, column=0, padx=15, pady=15, sticky='nsew')  # Fill the cell

        # Outer frame for model preview (scrollable)
        model_preview_frame_outer = ctk.CTkFrame(load_files_frame, corner_radius=10)
        model_preview_frame_outer.grid(row=2, column=0, padx=15, pady=15, sticky='nsew')  # Fill the cell

        # Scrollable frame for model preview
        model_preview_frame = ctk.CTkScrollableFrame(model_preview_frame_outer, corner_radius=10, fg_color="#353535", orientation=VERTICAL)
        model_preview_frame.pack(padx=15, pady=15, expand=True, fill='both')  # Fill the outer frame

        #data
        data_button = ctk.CTkButton(load_files_frame, text="Load Data (Excel/Parquet)",
                                font=self.button_font, 
                                command=lambda: load_data_file_and_preview(path_container, data_indicator, columns_frame))
        data_indicator = ctk.CTkLabel(load_files_frame, text="No Data Selected")
        data_button.grid(row=0, column = 1, padx=15, pady=15)
        data_indicator.grid(row = 1, column = 1, padx=15, pady=15)

        columns_frame = ctk.CTkScrollableFrame(load_files_frame, orientation=HORIZONTAL)
        columns_frame.grid(row=2, column=1, padx=15, pady=15, sticky="ew")
        
        configuration_metrics_frame = ctk.CTkFrame(self.main_frame,corner_radius=10)
        configuration_metrics_frame.pack(pady=15,side=TOP,fill=X,padx=20)
        configuration_metrics_frame.grid_columnconfigure(0, weight=1)
        configuration_metrics_frame.grid_columnconfigure(1, weight=1)

        hyperparameter_frame_label = ctk.CTkLabel(configuration_metrics_frame, text = "Training Configuration", font=self.button_font)
        hyperparameter_frame_label.grid(row=0,column=0,pady=15,padx=15)
        # Hyperparameter configuration section
        hyperparameter_frame = ctk.CTkFrame(configuration_metrics_frame, corner_radius=10)
        hyperparameter_frame.grid(row=1, column = 0, pady=15, padx=20, sticky='nsew')

        # Learning rate entry
        learning_rate_label = ctk.CTkLabel(hyperparameter_frame, text="Learning Rate:")
        learning_rate_label.grid(row=1, column=0, padx=15, pady=15)
        learning_rate_entry = ctk.CTkEntry(hyperparameter_frame)
        learning_rate_entry.grid(row=1, column=1, padx=15, pady=15)

        # Epochs entry
        epochs_label = ctk.CTkLabel(hyperparameter_frame, text="Epochs:")
        epochs_label.grid(row=2, column=0, padx=15, pady=15)
        epochs_entry = ctk.CTkEntry(hyperparameter_frame)
        epochs_entry.grid(row=2, column=1, padx=15, pady=5)

         # Optimiser
        optimiser_label = ctk.CTkLabel(hyperparameter_frame, text="Optimiser")
        optimiser_label.grid(row=1, column=3, padx=15, pady=15)
        optimiser_options = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
        optimiser_entry = ctk.CTkComboBox(hyperparameter_frame, values=optimiser_options)
        optimiser_entry.grid(row=1, column=4, padx=15, pady=15)
        optimiser_entry.set("Adam")  # Set default value
        # Loss
        loss_function_label = ctk.CTkLabel(hyperparameter_frame, text="Loss Function")
        loss_function_label.grid(row=2, column=3, padx=15, pady=15)
        loss_function_options = ['mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 
                                 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'poisson', 
                                 'kullback_leibler_divergence', 'hinge']
        loss_function_entry = ctk.CTkComboBox(hyperparameter_frame, values=loss_function_options)
        loss_function_entry.grid(row=2, column=4, padx=15, pady=15)
        loss_function_entry.set("mean_squared_error")  # Set default value

        # Metric
        metric_function_label = ctk.CTkLabel(hyperparameter_frame, text="Metrics")
        metric_function_label.grid(row=3, column=3, padx=15, pady=15)
        metric_function_options = ["accuracy", "binary_accuracy", "categorical_accuracy",
                                    "sparse_categorical_accuracy", "top_k_categorical_accuracy",
                                    "sparse_top_k_categorical_accuracy", "mean_squared_error",
                                    "root_mean_squared_error", "mean_absolute_error",
                                    "mean_absolute_percentage_error", "cosine_similarity",
                                    "hinge", "squared_hinge", "logcosh", "poisson"]
        metric_function_entry = ctk.CTkComboBox(hyperparameter_frame, values=metric_function_options)
        metric_function_entry.grid(row=3, column=4, padx=15, pady=15)
        metric_function_entry.set("mean_absolute_error")  # Set default value

        #metric frame
        training_metrics_label = ctk.CTkLabel(configuration_metrics_frame, text="Training Metrics", font=self.button_font)
        training_metrics_label.grid(row=0,column=1, padx=15, pady=20)
        training_metrics_frame = ctk.CTkFrame(configuration_metrics_frame, corner_radius=10)
        training_metrics_frame.grid(row=1, column=1, padx=15,pady=20, sticky='nsew')

        validation_loss_label = ctk.CTkLabel(training_metrics_frame, text="")
        validation_loss_label.grid(row=0,column=0, padx=15,pady=15)
        
        # Button to start model training
        train_model_button_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        train_model_button_frame.pack(pady=15,side=TOP,fill=X,padx=20)

        training_progressBar = ctk.CTkProgressBar(train_model_button_frame, orientation=HORIZONTAL, progress_color="#33cc33")
        training_progressBar.grid(row=0,column=1, padx=15,pady=15)
        training_progressBar.set(0)
        training_progressBar_label = ctk.CTkLabel(train_model_button_frame, text = "", font=self.button_font)
        training_progressBar_label.grid(row=0,column=2,padx=15,pady=15)
        
        saved_model_label = ctk.CTkLabel(train_model_button_frame, text="", font=self.button_font)
        saved_model_label.grid(row=0,column=1, padx=15,pady=15)

        train_model_button = ctk.CTkButton(
        train_model_button_frame, text="Start Training", font=self.button_font, 
        command=lambda: threading.Thread(
            target=start_training,
            args=(path_container, validation_loss_label, self.model_input_shape[1], self.model_input_shape[2], optimiser_entry.get(), loss_function_entry.get(), metric_function_entry.get(), learning_rate_entry.get(), epochs_entry.get(), saved_model_label, self, training_progressBar, training_progressBar_label),
            daemon=True  # Ensures the thread will exit when the main program does
        ).start()
    )
        train_model_button.grid(row=0,column=0,padx=15,pady=15)

 #------------------------------------------------------------------------------------------------------------------------------------------------
    def stop_trading(self):
        self.is_running = False

    def prepare_data_for_prediction(self, df, sequence_length_in, minutes_ahead):
        data = df.iloc[-(sequence_length_in + minutes_ahead):-minutes_ahead].copy()
        data = data[self.scaler_columns]
        data_scaled = self.scaler.transform(data)
        data_scaled_df = pd.DataFrame(data_scaled, columns=self.scaler_columns)
        X = data_scaled_df.values.reshape((1, sequence_length_in, len(self.scaler_columns)))
        return X

    def update_plot(self, minutes_ahead, selected_resolution, selected_crypto, labels):
        if not self.is_running:
            return

        try:
            # Fetch new data
            new_data = fetch_current_minute_data(symbol=selected_crypto, interval=selected_resolution)
            if new_data.empty:
                print("No new data fetched.")
            else:
                last_actual_index = -minutes_ahead - 1
                new_timestamp = new_data.index[0]
                if new_timestamp > self.df_full.index[last_actual_index]:
                    last_index = self.df_full.index[-1]
                    next_index = last_index + pd.Timedelta(minutes=1)
                    buffer_row = pd.DataFrame([[np.nan] * len(self.df_full.columns)], columns=self.df_full.columns, index=[next_index])
                    self.df_full = pd.concat([self.df_full, buffer_row])
                    self.df_full.loc[self.df_full.index[last_actual_index], ['Open', 'High', 'Low', 'Close', 'Volume']] = new_data.iloc[0]

                prediction_time = self.df_full.index[-1]
                prediction_price = None

                # Check if a prediction is already made for this time
                if np.isnan(self.df_full.loc[prediction_time, 'Prediction']):
                    if self.model is not None and self.scaler is not None and self.scaler_columns is not None:
                        # Get the model input shape
                        _, sequence_length_in, num_features = self.model.input_shape
                        
                        # Prepare data for prediction
                        X = self.prepare_data_for_prediction(self.df_full, sequence_length_in - minutes_ahead, minutes_ahead)
                        if len(X) > 0:
                            # Make prediction
                            prediction = self.model.predict(X)
                            inverse_transform_array = np.zeros((1, len(self.scaler_columns)))
                            inverse_transform_array[0, 0] = prediction
                            prediction_price = self.scaler.inverse_transform(inverse_transform_array)[0, 0]

                """
                Trading Strategy Implementation Section
                ---------------------------------------
                This is the section where you implement your trading strategy.
                Based on the model's prediction, you can decide whether to buy, sell, or hold.
                Example:
                if prediction_price > current_price + threshold:
                    execute_buy_order()
                elif prediction_price < current_price - threshold:
                    execute_sell_order()
                else:
                    hold_position()

                Replace the following line with your trading logic.
                """
                # For testing: doesn't actually represent real prediction
                prediction_price = self.df_full.iloc[-minutes_ahead - 1]['Close']

                # Update plot with new data and prediction
                plot_data(self.df_full, [self.ax1, self.ax2], self.canvas, prediction_time, prediction_price, minutes_ahead)

                # Update UI labels
                labels['current_time_label'].configure(text=f"Current Time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
                labels['selected_crypto_label'].configure(text=f"Selected Crypto: {selected_crypto}")
                labels['selected_resolution_label'].configure(text=f"Selected Resolution: {selected_resolution}")

                candlestick_data = self.df_full.iloc[last_actual_index]
                candlestick_text = (
                    f"Open: {candlestick_data['Open']}\n"
                    f"High: {candlestick_data['High']}\n"
                    f"Low: {candlestick_data['Low']}\n"
                    f"Close: {candlestick_data['Close']}\n"
                    f"Volume: {candlestick_data['Volume']}"
                )
                labels['most_recent_candlestick_label'].configure(text=f"Most Recent Candlestick:\n{candlestick_text}")

                if prediction_price is not None:
                    labels['most_recent_prediction_label'].configure(text=f"Most Recent Prediction: {prediction_price:.3f}")

        except Exception as e:
            print(f"Error in update_plot: {e}")

        self.after(1000, lambda: self.update_plot(minutes_ahead, selected_resolution, selected_crypto, labels))

    def initialise_trading(self, frame, selected_resolution, selected_crypto, path_container, minutes_ahead, labels):
        # Load the model
        self.model = load_model(path_container['model_path'])
        # Load the scaler and columns
        self.scaler = joblib.load(path_container['scaler_path'])
        self.scaler_columns = joblib.load(path_container['columns_path'])

        # Initialize the data and plot
        self.is_running = True
        self.df_full = fetch_data(symbol=selected_crypto, interval=selected_resolution)
        self.df_full = extend_future_data(self.df_full, minutes_ahead)

        self.fig, self.axes = mpf.plot(
            self.df_full,
            type='candle',
            style='yahoo',
            returnfig=True,
            figscale=1,
            figratio=(16, 9),
            title="Live Trading",
            scale_padding={'left': 0.1, 'right': 0.70, 'top': 0.4, 'bottom': 0.7},
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.update_plot(minutes_ahead, selected_resolution, selected_crypto, labels)

    def select_resolution(self, resolution):
        self.selected_resolution = resolution
        self.update_resolution_buttons()

    def update_resolution_buttons(self):
        for res, button in self.resolution_buttons.items():
            if res == self.selected_resolution:
                button.configure(fg_color="#45b057")
            else:
                button.configure(fg_color="#3a7ebf")

    def load_realtime(self):
        self.is_running = False
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        path_container = {}

        realTime_title = ctk.CTkLabel(self.main_frame, text="Real-Time Trading", font=self.title_font, text_color="#353535")
        realTime_title.pack(pady=20, padx=25, side=TOP, anchor="w")

        loader_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        loader_frame.pack(pady=15, side=TOP, fill=X, padx=20)

        model_button = ctk.CTkButton(loader_frame, text="Load Model (.keras)", font=self.button_font, command=lambda: load_model_file(path_container, model_indicator))
        model_indicator = ctk.CTkLabel(loader_frame, text="No Model Selected")
        scaler_button = ctk.CTkButton(loader_frame, text="Load Scaler (.pkl)", font=self.button_font, command=lambda: load_scaler_file(path_container, scaler_indicator))
        scaler_indicator = ctk.CTkLabel(loader_frame, text="No Scaler Model Selected")
        columns_button = ctk.CTkButton(loader_frame, text="Load Columns (.pkl)", font=self.button_font, command=lambda: load_columns_file(path_container, columns_indicator))
        columns_indicator = ctk.CTkLabel(loader_frame, text="No Columns File Selected")

        model_button.grid(row=0, column=0, padx=15, pady=15)
        model_indicator.grid(row=1, column=0, padx=15, pady=15)
        scaler_button.grid(row=0, column=1, padx=15, pady=15)
        scaler_indicator.grid(row=1, column=1, padx=15, pady=15)
        columns_button.grid(row=0, column=2, padx=15, pady=15)
        columns_indicator.grid(row=1, column=2, padx=15, pady=15)

        self.selected_resolution = '1m'

        config_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        config_frame.pack(pady=15, side=TOP, fill=X, padx=20)

        select_crypto_label = ctk.CTkLabel(config_frame, text="Select Crypto", font=self.button_font)
        select_crypto_label.grid(row=0, column=0, padx=15, pady=15)
        select_crypto_input = ctk.CTkEntry(config_frame, corner_radius=10)
        select_crypto_input.grid(row=0, column=1, padx=15, pady=15)

        resolutions = ['1m', '5m', '15m', '30m', '1h']
        self.resolution_buttons = {}
        for i, resolution in enumerate(resolutions):
            button = ctk.CTkButton(config_frame, text=resolution, font=self.button_font, command=lambda res=resolution: self.select_resolution(res), width=50)
            button.grid(row=0, column=i + 2, padx=5, pady=5)
            self.resolution_buttons[resolution] = button

        self.update_resolution_buttons()

        minutes_ahead_label = ctk.CTkLabel(config_frame, text="Minutes Ahead to Predict", font=self.button_font)
        minutes_ahead_label.grid(row=0, column=7, padx=15, pady=15)
        minutes_ahead_input = ctk.CTkEntry(config_frame, corner_radius=10)
        minutes_ahead_input.grid(row=0, column=8, padx=15, pady=15)

        graph_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        graph_frame.pack(pady=15, side=TOP, fill=X, padx=20)
        graph_inner_frame = ctk.CTkFrame(graph_frame, corner_radius=10)
        graph_inner_frame.grid(row=0, column=0, pady=15, padx=15, sticky="nsew")
        graph_sidebar_frame = ctk.CTkFrame(graph_frame, corner_radius=10)
        graph_sidebar_frame.grid(row=0, column=1, pady=15, padx=15, sticky="nsew")
        graph_frame.grid_columnconfigure(0, weight=15)
        graph_frame.grid_columnconfigure(1, weight=1)

        graph_sidebar_details_frame = ctk.CTkFrame(graph_sidebar_frame, corner_radius=10)
        graph_sidebar_details_frame.pack(pady=15, padx=15, fill=BOTH)

        labels = {
            'selected_crypto_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Selected Crypto"),
            'current_time_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Current Time"),
            'selected_resolution_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Selected Resolution"),
            'most_recent_candlestick_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Most Recent Candlestick"),
            'most_recent_prediction_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Most Recent Prediction"),
            'starting_equity_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Starting Equity: 100000"),
            'current_equity_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Current Equity: 100000"),
            'return_so_far_label': ctk.CTkLabel(graph_sidebar_details_frame, font=self.button_font, text="Return So Far: 0")
        }

        for label in labels.values():
            label.pack(pady=15, padx=15)

        start_trading_button = ctk.CTkButton(config_frame, corner_radius=10, text="Start Trading", command=lambda: self.initialise_trading(graph_inner_frame, self.selected_resolution, select_crypto_input.get(), path_container, int(minutes_ahead_input.get()), labels))
        start_trading_button.grid(row=1, column=0, padx=15, pady=15)
        
 #------------------------------------------------------------------------------------------------------------------------------------------------


        


