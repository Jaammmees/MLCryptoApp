import customtkinter as ctk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
#from trading.historical import run_backtest_and_save_report
import tkinterweb
import threading
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
import tensorflow as tf
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

from backtesting.test import EURUSD

class MainWindow(ctk.CTk):
    def __init__(self, *args, **kwargs):
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

        self.setup_sidebar()

        self.load_historical()

        self.create_widgets()

    def create_widgets(self):
        pass

    def display_model_summary(self, model_path, display_frame):
        """
        Loads a .h5 model, captures its summary, and displays it in a provided customtkinter frame.
        """
        # Load the model and capture its summary
        model = load_model(model_path)
        str_io = io.StringIO()

        with redirect_stdout(str_io):
            model.summary()

        summary_lines = str_io.getvalue().splitlines()

        # Parse the summary
        header, rows, footer = self.parse_summary(summary_lines)

        # Clear previous widgets from the frame
        for widget in display_frame.winfo_children():
            widget.destroy()

        # Display the new model summary in the frame
        for line in header + rows + footer:
            label = ctk.CTkLabel(master=display_frame, text=line, text_color='#FFFFFF', font=self.button_font)
            label.pack(pady=2, padx=10, anchor="w")

        display_frame.pack(padx=15, pady=15, expand=True, fill='both')

    def parse_summary(self, lines):
        header = lines[0:4]  # Assuming first 4 lines include the headers and separators
        rows = lines[4:-5]  # The model layers are listed after the header and before the total params
        footer = lines[-5:]  # Total params and other details
        return header, rows, footer

    def setup_sidebar(self):

        #images
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
        
        #buttons
        button_backtest = ctk.CTkButton(self.sidebar, text="Backtest", font=self.navbar_font, compound=BOTTOM, command=self.load_historical, image=self.backtestImage, width=60,height=60)
        button_realtime = ctk.CTkButton(self.sidebar, text="Real-Time", font=self.navbar_font, compound=BOTTOM, command=self.load_realtime, image=self.realTimeImage, width=60,height=60)
        button_build = ctk.CTkButton(self.sidebar, text="Build Model", font=self.navbar_font, compound=BOTTOM, command=self.load_build_model, image=self.buildImage, width=60,height=60)
        button_data = ctk.CTkButton(self.sidebar, text="Process Data", font=self.navbar_font, compound=BOTTOM, command=self.load_process_data, image=self.dataImage, width=60,height=60)
        button_training = ctk.CTkButton(self.sidebar, text="Train Model", font=self.navbar_font, compound=BOTTOM, command=self.load_train_model, image=self.trainImage, width=60,height=60)
        
        #grid positioning
        button_backtest.grid(row=1, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_realtime.grid(row=2, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_build.grid(row=3, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_data.grid(row=4, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_training.grid(row=5, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")

        self.sidebar.grid_columnconfigure(0, weight=1)  # Make column 0 take up all available space
        self.sidebar.grid_rowconfigure(0, weight=1)  # Spacer row at the top
        self.sidebar.grid_rowconfigure(1, weight=0)  # Actual button row
        self.sidebar.grid_rowconfigure(2, weight=0)  # Actual button row
        self.sidebar.grid_rowconfigure(3, weight=0)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(4, weight=0)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(5, weight=0)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(6, weight=1)  # Spacer row at the bottom



    def load_historical(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        path_container = {}

        #internal functions to laod files
        def load_model_file():
            model_file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
            if model_file_path:
                #print("Model loaded:", model_file_path)
                model_indicator.configure(text="Model " + os.path.basename(model_file_path) + " loaded")
                path_container['model_path'] = model_file_path

        def load_scaler_file():
            scaler_file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
            if scaler_file_path:
                #print("Scaler loaded:", scaler_file_path)
                scaler_indicator.configure(text="Model Scaler " + os.path.basename(scaler_file_path) + " loaded")
                path_container['scaler_path'] = scaler_file_path 

        def load_data_file():
            data_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("Parquet files", "*.parquet")])
            if data_file_path:
                #print("Data loaded:", data_file_path)
                data_indicator.configure(text="Data " + os.path.basename(data_file_path) + " loaded")
                path_container['data_path'] = data_file_path

        #--------------------------------------------------------------------------------------------------

        #title
        backtest_title = ctk.CTkLabel(self.main_frame, text="Backtesting", font=self.title_font, text_color="#353535")
        backtest_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        #loading frame
        loader_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        loader_frame.pack(pady=5, side=TOP, fill = X, padx = 20)

        #need to be able to load a machine model, and a scalar, choose the data we want to backtest it on LOCALLY or just a TICKER that we can grab
        #the data has to be in the form INDEX == Timestep, Open, High, Low, Close, Volume as of now
        #buttons and text for loading model, scaler, and data
        model_button = ctk.CTkButton(loader_frame, text="Load Model (.h5)", font=self.button_font, command=load_model_file)
        model_indicator = ctk.CTkLabel(loader_frame, text="No Model Selected")
        scaler_button = ctk.CTkButton(loader_frame, text="Load Scaler (.pkl)", font=self.button_font, command=load_scaler_file)
        scaler_indicator = ctk.CTkLabel(loader_frame, text="No Scaler Model Selected")
        data_button = ctk.CTkButton(loader_frame, text="Load Data (Excel/Parquet)", font=self.button_font, command=load_data_file)
        data_indicator = ctk.CTkLabel(loader_frame, text="No Data Selected")

        #pack buttons side by side
        model_button.grid(row=0, column = 0, padx=15, pady=15)
        model_indicator.grid(row = 1, column = 0, padx=15, pady=15)
        scaler_button.grid(row=0, column = 1, padx=15, pady=15)
        scaler_indicator.grid(row = 1, column = 1, padx=15, pady=15)
        data_button.grid(row=0, column = 2, padx=15, pady=15)
        data_indicator.grid(row = 1, column = 2, padx=15, pady=15)

        #--------------------------------------------------------------------------------------------------
        
        #all backtesting functions

        # def backtest_button_callback():
        #     # Load data and create strategy instance here, or ensure they are accessible
        #     data = EURUSD.copy()
            
        #     # Run backtest in a separate thread to avoid freezing the GUI
        #     thread = threading.Thread(target=lambda: run_backtest_and_update_gui(data))
        #     thread.start()

        # def run_backtest_and_update_gui(data):
        #     results, report_path = run_backtest_and_save_report(data)
        #     display_backtest_results(backtest_result_frame, report_path, results)
        
        # def display_backtest_results(master, html_file, results):
        #     # Clear previous results if necessary
        #     for widget in master.winfo_children():
        #         widget.destroy()

        #     # Frame for the web view
        #     web_frame = ctk.CTkFrame(master)
        #     web_frame.pack(fill='both', expand=True)

        #     # Display the HTML report
        #     html_view = tkinterweb.HtmlFrame(web_frame, horizontal_scrollbar="auto")
        #     html_view.load_file(html_file)  # Ensure html_file is the correct, full path
        #     html_view.pack(fill='both', expand=True)

        #     # Optionally display the results in a text widget or labels
        #     results_text = ctk.CTkLabel(master, text=str(results))
        #     results_text.pack(pady=20)


        #-----------------------------------------------------------------------
        
        #backtesting frame
        backtest_button_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        backtest_button_frame.pack(pady=20, side=TOP, fill = X, padx = 20)

        #have a button that hits backtest, which calls the backtest function, which will return a html and we will embed that,
                                                                                                        #comamnd missing here
        backtest_button = ctk.CTkButton(backtest_button_frame, text="Run Backtest", font=self.button_font)
        backtest_button.grid(row=0, column=0, padx=15, pady=15)


        #backtest results frame

        backtest_result_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        backtest_result_frame.pack(pady=10, side=TOP, fill = BOTH, padx = 20)

        #in relation to building the model
    def update_layer_type(self, value, index):
        # Update the layer type at specific index
        self.layer_widgets[index]['type'].set(value)

        # Show or hide the return sequences checkbox based on the layer type
        if value == 'LSTM':
            self.layer_widgets[index]['return_seq'].grid()
        else:
            self.layer_widgets[index]['return_seq'].grid_remove()

        #in relation to building the model
    def update_layer_param(self, event, index):
            entry_widget = event.widget
            current_text = entry_widget.get()  # This gets the current text from the entry
            #print(current_text)
            self.layer_widgets[index]['params'] = int(current_text)
            #print(self.layer_widgets)

    def load_build_model(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        #internal functions
        #updates layers

        def update_layer_widgets(self, num_layers):
            # Clear existing widgets
            for widget in self.building_layers_frame.winfo_children():
                widget.destroy()

            self.layer_widgets = []
            for i in range(num_layers):
                layer_label = ctk.CTkLabel(self.building_layers_frame, text=f"Layer {i + 1}:")
                layer_label.grid(row=i, column=0, padx=15, pady=15)
                
                layer_type_combo = ctk.CTkComboBox(self.building_layers_frame, values=["", "LSTM", "Dense", "Dropout"])
                layer_type_combo.grid(row=i, column=1, padx=15, pady=15)

                layer_param_entry = ctk.CTkEntry(self.building_layers_frame)
                layer_param_entry.grid(row=i, column=2, padx=15, pady=15)

                layer_return_seq_chk = ctk.CTkCheckBox(self.building_layers_frame, text="Return Sequences")
                layer_return_seq_chk.grid(row=i, column=3, padx=15, pady=15)
                layer_return_seq_chk.grid_remove()  # Hide initially
                
                self.layer_widgets.append({
                    'type': layer_type_combo, 
                    'params': layer_param_entry, 
                    'return_seq': layer_return_seq_chk
                })

                # Set callback to update the type in the list when changed
                layer_type_combo.configure(command=lambda value, idx=i: self.update_layer_type(value, idx))
                layer_param_entry.bind('<KeyRelease>', lambda event, idx=i: self.update_layer_param(event, idx))

        #slider
        def slider_value(value):
            int_value = int(value)
            select_layers_value.configure(text=f"{int_value}")
            update_layer_widgets(self,int_value)

        #build model
        def build_model(layer_info, input_shape):
            model = Sequential()
            print(layer_info, input_shape)
            model.add(Input(shape=input_shape))
            for info in layer_info:
                layer_type = info['type']
                if layer_type == 'LSTM':
                    model.add(LSTM(info['params'], return_sequences=info['return_sequences']))
                elif layer_type == 'Dropout':
                    model.add(Dropout(info['params']))
                elif layer_type == 'Dense':
                    model.add(Dense(info['params']))

            return model
        
        def collect_model_details():
            layer_details = []
            #print(self.layer_widgets)
            for layer_info in self.layer_widgets:
                layer_type = layer_info['type'].get()
                layer_params = int(layer_info['params'])
                return_seq = layer_info['return_seq'].get() if layer_type == 'LSTM' else False
                layer_details.append({
                    'type': layer_type,
                    'params': layer_params,
                    'return_sequences': return_seq
                })

            hyperparameters = {
                'sequence_length_in': sequence_length_in_entry.get(),
                'sequence_length_out': sequence_length_out_entry.get(),
                'optimiser' : optimiser_entry.get(),
                'loss' : loss_function_entry.get(),
                'model name' : model_name_entry.get()
            }

            return layer_details, hyperparameters
        
        def build_and_save_model():
            layer_info, hyper_params = collect_model_details()
            # Assume input_shape is derived from 'sequence_length_in' and 'sequence_length_out'
            input_shape = (int(hyper_params['sequence_length_in']), int(hyper_params['sequence_length_out']))
            print(layer_info, hyper_params, input_shape)
            model = build_model(layer_info, input_shape)
            model.compile(optimizer=hyper_params['optimiser'], loss=hyper_params['loss'])  # Example configuration
            model.save('models/' + hyper_params['model name'] + '.h5')  # Save the model

            model_path = 'models/' + hyper_params['model name'] + '.h5'
            self.display_model_summary(model_path, self.model_preview_frame)

        #title
        build_model_title = ctk.CTkLabel(self.main_frame, text="Build Model", font=self.title_font, text_color="#353535")
        build_model_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        #select model frame
        select_model_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        select_model_frame.pack(pady=15, side=TOP, fill = X, padx = 20)

        # Model Type Selection: Dropdown to choose between different types of models (e.g., LSTM, CNN).
        select_model_combo = ctk.CTkComboBox(select_model_frame, values = ["LSTM", "CNN", "RNN"], height = 50, width = 150, font=self.combo_box_font)
        select_model_combo.grid(row=0,column=0, padx=15,pady=15, ipadx=30)
        #choose how many layers wanted
        select_layers_text = ctk.CTkLabel(select_model_frame, text="Model Layers")
        select_layers_text.grid(row=0,column=3,padx=15,pady=15)

        # Model Name
        model_name_label = ctk.CTkLabel(select_model_frame, text="Model Name")
        model_name_label.grid(row=0, column=1, padx=15, pady=15)
        model_name_entry = ctk.CTkEntry(select_model_frame)
        model_name_entry.grid(row=0, column=2, padx=15, pady=15)

        #slider
        select_layers = ctk.CTkSlider(select_model_frame, from_= 1, to = 10, number_of_steps=10, command=slider_value)
        select_layers.set(1)
        select_layers.grid(row=0,column=4, padx=15,pady=15, ipadx=30)
        select_layers_value = ctk.CTkLabel(select_model_frame, text="1")
        select_layers_value.grid(row=0,column=5,padx=15,pady=15)

        #choosing layers and their values
        #building layers frame
        self.building_layers_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        self.building_layers_frame.pack(pady=15, side=TOP, fill = X, padx = 20)

        update_layer_widgets(self,1)

        # Hyperparameters:
        parameters_title = ctk.CTkLabel(self.main_frame, text = "Hyperparameter Configuration & Model Preview", font = self.combo_box_font, text_color="#353535")
        parameters_title.pack(padx=15,pady=15, side=TOP, anchor = "w")

        side_by_side_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        side_by_side_frame.pack(pady=20, side=TOP, fill=X, padx=20)

        # Dropdowns for activation functions and optimizer choices.
        select_parameters_frame = ctk.CTkFrame(side_by_side_frame, corner_radius= 10, fg_color='transparent', bg_color='transparent')
        select_parameters_frame.pack(pady=15, side=LEFT, fill = X, padx = 20)
        
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
        # Optimiser
        optimiser_label = ctk.CTkLabel(select_parameters_frame, text="Optimiser")
        optimiser_label.grid(row=1, column=2, padx=15, pady=15)
        optimiser_options = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
        optimiser_entry = ctk.CTkComboBox(select_parameters_frame, values=optimiser_options)
        optimiser_entry.grid(row=1, column=3, padx=15, pady=15)
        optimiser_entry.set("Adam")  # Set default value
        # Loss
        loss_function_label = ctk.CTkLabel(select_parameters_frame, text="Loss Function")
        loss_function_label.grid(row=2, column=2, padx=15, pady=15)
        loss_function_options = ['mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 
                                 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'poisson', 
                                 'kullback_leibler_divergence', 'hinge']
        loss_function_entry = ctk.CTkComboBox(select_parameters_frame, values=loss_function_options)
        loss_function_entry.grid(row=2, column=3, padx=15, pady=15)
        loss_function_entry.set("mean_squared_error")  # Set default value

        # Submission: Button to confirm the setup and proceed to data preparation.
        model_build_button_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        model_build_button_frame.pack(pady=20, side=TOP, fill = X, padx = 20)

        #have a button that hits backtest, which calls the backtest function, which will return a html and we will embed that,
        model_build_button = ctk.CTkButton(model_build_button_frame, text="Build Model", font=self.button_font, command=lambda: build_and_save_model())
        model_build_button.grid(row=0, column=0, padx=15, pady=15)

        #model preview
        self.model_preview_frame = ctk.CTkFrame(side_by_side_frame, corner_radius= 10, fg_color='#333333')
        self.model_preview_frame.pack(pady=20, side=RIGHT, fill = X, padx = 20)
        self.model_preview_frame.pack_forget() 



    def load_process_data(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        #internal functions
        self.columns = {}
        path_container = {}

        def load_model_file():
            model_file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
            if model_file_path:
                #print("Model loaded:", model_file_path)
                model_indicator.configure(text="Model " + os.path.basename(model_file_path) + " loaded")
                path_container['model_path'] = model_file_path
                self.display_model_summary(model_file_path, model_preview_frame)

        def load_data_file(self):
            data_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Parquet files", "*.parquet")])
            path_container['data_path'] = data_file_path
            if data_file_path:
                data_indicator.configure(text="Data " + os.path.basename(data_file_path) + " loaded")
                try:
                    if data_file_path.endswith('.csv'):
                        data = pd.read_csv(data_file_path)
                    elif data_file_path.endswith('.xlsx'):
                        data = pd.read_excel(data_file_path)
                    elif data_file_path.endswith('.parquet'):
                        data = pd.read_parquet(data_file_path)

                    # Clear previous data in the frame
                    for widget in columns_frame.winfo_children():
                        widget.destroy()

                    for widget in self.column_list_frame.winfo_children():
                        widget.destroy()
                    
                    # Create labels for column names and first few rows
                    for i, column in enumerate(data.columns):
                        ctk.CTkLabel(columns_frame, text=column, width=20).grid(row=0, column=i)
                
                        frame = ctk.CTkFrame(self.column_list_frame, height = 50)
                        frame.pack(side="left", fill="x", expand=True,padx=10,pady=10)

                        label = ctk.CTkLabel(frame, text=column)
                        label.pack(side="left", fill="x", expand=True, padx=10)

                        delete_button = ctk.CTkButton(frame, text="X", width=40, height=40, command=lambda c=column: delete_column(self,c))
                        delete_button.pack(side="right")

                        self.columns[column] = frame

                    for row_index in range(min(5, len(data))):
                        for col_index, column in enumerate(data.columns):
                            ctk.CTkLabel(columns_frame, text=str(data.iloc[row_index, col_index]), width=60).grid(row=row_index + 1, column=col_index)

                except Exception as e:
                    data_indicator.configure(text=f"Failed to load data: {str(e)}")

        def delete_column(self, column):
            if column in self.columns:
                self.columns[column].destroy()  # Remove the widget
                del self.columns[column]  # Remove the reference

        def upload_scaler_prompt(value, frame):

            def load_scaler_file():
                scaler_file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
                if scaler_file_path:
                    #print("Scaler loaded:", scaler_file_path)
                    scaler_indicator.configure(text="Model Scaler " + os.path.basename(scaler_file_path) + " loaded")
                    path_container['scaler_path'] = scaler_file_path 

            scaler_button = ctk.CTkButton(frame, text="Load Scaler (.pkl)", font=self.button_font, command=load_scaler_file)
            scaler_indicator = ctk.CTkLabel(frame, text="No Scaler Model Selected")

            if value == "Upload Scaler":
                scaler_button.grid(row=0, column=2,padx=15,pady=15)
                scaler_indicator.grid(row=0,column=3,padx=15,pady=15)
            else:
                scaler_button.grid_forget()
                scaler_indicator.grid_forget()

        def display_data_preview(self, scaled_data):
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

            # Configure the frame to be scrollable if it's not already
            data_preview_frame.pack(side="left", fill="both", expand=True)

        def process_and_save_data(self):
            try:
                original_data_path = path_container['data_path']
                if not original_data_path:
                    raise Exception("Data not loaded")
                
                data = pd.read_csv(original_data_path) if original_data_path.endswith('.csv') \
                    else pd.read_excel(original_data_path) if original_data_path.endswith('.xlsx') \
                    else pd.read_parquet(original_data_path)

                # Step 2: Delete columns not present in self.columns
                current_columns = list(self.columns.keys())
                data = data[current_columns]

                # Step 3: Apply scaler
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
                display_data_preview(self,scaled_data)  # Display the data in a simplified format

            except Exception as e:
                error_label = ctk.CTkLabel(data_preview_frame, text="Error: " + str(e), fg_color="#FFFFFF")
                error_label.pack(pady=15, padx=15, fill='both', expand=True)

        process_data_title = ctk.CTkLabel(self.main_frame, text="Process Data", font=self.title_font, text_color="#353535")
        process_data_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        #Data Upload:
        # Button to upload dataset files.
        # Display of uploaded file paths to confirm the data is loaded.
        # allow for loading of model (to see input and output shape) and allow for loading of data to be processed
        # Frame to hold file loading components
        load_files_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        load_files_frame.pack(pady=15, side='top', fill='x', padx=20)
        load_files_frame.grid_columnconfigure(0, weight=1)  # Set equal weight if needed
        load_files_frame.grid_columnconfigure(1, weight=1)  # Set equal weight if needed

        # Button to load a model
        model_button = ctk.CTkButton(load_files_frame, text="Load Model (.h5)", font=self.button_font, command=load_model_file)
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
        data_button = ctk.CTkButton(load_files_frame, text="Load Data (Excel/Parquet)", font=self.button_font, command=lambda: load_data_file(self))
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
        choose_scaler_combo = ctk.CTkComboBox(normalisation_frame, values=['MinMaxScaler', 'StandardScalar', 'Upload Scaler'], command=lambda value : upload_scaler_prompt(value, normalisation_frame))
        choose_scaler_combo.grid(row=0,column=1,padx=15,pady=15)

        # Sequence Preparation:
        # intake the model shape for the sequence length, crucial for LSTM models.
        data_process_button_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        data_process_button_frame.pack(pady=15,side=TOP,fill=X,padx=20)

        data_process_button = ctk.CTkButton(data_process_button_frame, font=self.button_font, text="Process Data & Save", command=lambda: process_and_save_data(self))
        data_process_button.grid(row=0,column=0,padx=15,pady=15)

        data_preview_frame_outer = ctk.CTkFrame(self.main_frame,corner_radius=10)
        data_preview_frame_outer.pack(pady=15,padx=20, side=TOP,fill=X)

        processed_data_indicator = ctk.CTkLabel(data_preview_frame_outer, text=f"Preview of processed data", font=self.button_font)
        processed_data_indicator.pack(pady=15,padx=15,side=TOP,fill=X)

        data_preview_frame = ctk.CTkScrollableFrame(data_preview_frame_outer,corner_radius=10, fg_color="#353535")
        data_preview_frame.pack(pady=15,padx=15,side=TOP,fill=X)


        
    #------------------------------------------------------------------------------------------------------------------------------------------------

    def load_train_model(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        process_data_title = ctk.CTkLabel(self.main_frame, text="Model Training", font=self.title_font, text_color="#353535")
        process_data_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        path_container = {}
        self.model_input_shape = None
        self.model_output_shape = None
        #internal functions
        def load_model_file():
            model_file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
            if model_file_path:
                #print("Model loaded:", model_file_path)
                model_indicator.configure(text="Model " + os.path.basename(model_file_path) + " loaded")
                path_container['model_path'] = model_file_path
                model = load_model(path_container['model_path'])
                self.model_input_shape = model.input_shape
                self.model_output_shape = model.output_shape
                model_shape.configure(text=f"Shape of Model Input is: {model.input_shape}, \n and Model Output is: {model.output_shape}")
                model_shape.grid()

        def load_data_file():
            data_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("Parquet files", "*.parquet"), ("CSV files", "*.csv")])
            if data_file_path:
                #print("Data loaded:", data_file_path)
                data_indicator.configure(text="Data " + os.path.basename(data_file_path) + " loaded")
                path_container['data_path'] = data_file_path

        def create_sequences(data, sequence_length, prediction_steps, target_column='Close', include_columns=None):
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

        def generate_sequence(input_shape):
            # Validate sequence length entry
            if not sequence_length_entry.get().isdigit():
                sequence_preview_text.configure(text="Invalid sequence length.")
                return
            if 'data_path' not in path_container:
                sequence_preview_text.configure(text="No data file loaded.")
                return
            print(input_shape)
            sequence_length = int(sequence_length_entry.get())
            if sequence_length != input_shape[1]:  # assuming input_shape is like (None, sequence_length, num_features)
                sequence_preview_text.configure(text=f"Sequence length mismatch. Model expects {input_shape[1]}.")
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
                sequence_preview_text.configure(text="Sequence generated successfully. Number of sequences: " + str(len(X)))
            except Exception as e:
                sequence_preview_text.configure(text=f"Error: {str(e)}")


        def start_training():
            if 'model_path' not in path_container or 'data_path' not in path_container:
                sequence_preview_text.configure(text="Model or data file not loaded.")
                return

            # Load model
            model = load_model(path_container['model_path'])

            # Fetch hyperparameters
            try:
                learning_rate = float(learning_rate_entry.get())
                epochs = int(epochs_entry.get())
            except ValueError:
                sequence_preview_text.configure(text="Invalid hyperparameters.")
                return

            # Load and prepare data
            data = pd.read_csv(path_container['data_path'])
            sequence_length = int(sequence_length_entry.get())  # Ensure this entry is validated earlier in the workflow
            X, y = create_sequences(data, sequence_length, prediction_steps=5, target_column='close')

            # Split data into training and testing
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='mean_squared_error',
                        metrics=['mean_absolute_error'])

            # Train the model
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

            # Update GUI post-training
            sequence_preview_text.configure(text=f"Training complete. Final validation loss: {history.history['val_loss'][-1]}")

            # Optionally save the trained model
            model.save('updated_model.h5')

        load_files_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        load_files_frame.pack(pady=15,side=TOP,fill=X,padx=20)

        #model stuff
        model_button = ctk.CTkButton(load_files_frame, text="Load Model (.h5)", font=self.button_font, command=load_model_file)
        model_indicator = ctk.CTkLabel(load_files_frame, text="No Model Selected")
        model_button.grid(row=0, column = 0, padx=15, pady=15)
        model_indicator.grid(row = 1, column = 0, padx=15, pady=15)
        model_shape = ctk.CTkLabel(load_files_frame, text="")
        model_shape.grid(row=2, column = 0, padx=15,pady=15)
        model_shape.grid_remove()

        #data stuff
        data_button = ctk.CTkButton(load_files_frame, text="Load Data (Excel/Parquet)", font=self.button_font, command=load_data_file)
        data_indicator = ctk.CTkLabel(load_files_frame, text="No Data Selected")
        data_button.grid(row=0, column = 2, padx=15, pady=15)
        data_indicator.grid(row = 1, column = 2, padx=15, pady=15)

        #sequence generation
        sequence_generation_frame = ctk.CTkFrame(self.main_frame,corner_radius=10)
        sequence_generation_frame.pack(pady=15,side=TOP,fill=X,padx=20)

        # Sequence generation setup
        sequence_label = ctk.CTkLabel(sequence_generation_frame, text="Configure Sequence Generation", font=self.button_font)
        sequence_label.grid(row=0, column=0, padx=15, pady=5)

        # Sequence length entry
        sequence_length_label = ctk.CTkLabel(sequence_generation_frame, text="Sequence Length:")
        sequence_length_label.grid(row=1, column=0, padx=15, pady=15)
        sequence_length_entry = ctk.CTkEntry(sequence_generation_frame)
        sequence_length_entry.grid(row=1, column=1, padx=15, pady=15)

        # Button to apply sequence settings and preview the sequence
        sequence_apply_button = ctk.CTkButton(sequence_generation_frame, text="Generate Sequence", command=lambda: generate_sequence(self.model_input_shape))
        sequence_apply_button.grid(row=2, column=1, padx=15, pady=15)

        # Placeholder for sequence preview
        sequence_preview_label = ctk.CTkLabel(sequence_generation_frame, text="Sequence Preview:")
        sequence_preview_label.grid(row=3, column=0, padx=15, pady=15)
        sequence_preview_text = ctk.CTkLabel(sequence_generation_frame, text="", font=("Helvetica", 10))
        sequence_preview_text.grid(row=3, column=1, padx=15, pady=15)

        # Hyperparameter configuration section
        hyperparameter_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        hyperparameter_frame.pack(pady=15, side=TOP, fill=X, padx=20)
        hyperparameter_label = ctk.CTkLabel(sequence_generation_frame, text="Configure Sequence Generation", font=self.button_font)
        hyperparameter_label.grid(row=0, column=0, padx=15, pady=5)
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

        # Button to start model training
        train_model_button = ctk.CTkButton(hyperparameter_frame, text="Start Training", command=start_training)
        train_model_button.grid(row=3, column=1, padx=15, pady=15)




    def load_realtime(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(self.main_frame, text = "realtime area")
        label.pack(pady=20)


    
        


if __name__ == "__main__":
    app = MainWindow
    app.mainloop()