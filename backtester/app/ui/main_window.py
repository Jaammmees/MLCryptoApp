import customtkinter as ctk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
#from trading.historical import run_backtest_and_save_report
import tkinterweb
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


from backtesting.test import EURUSD

class MainWindow(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title('Machine Learning Backtester')
        self.geometry('1600x900')

        self._set_appearance_mode("light")
        self._apply_appearance_mode("blue")
        

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

    def setup_sidebar(self):

        #images
        chart_image = Image.open("./images/line-chart-svgrepo-com.png")
        dollar_image = Image.open("./images/dollar-sign-svgrepo-com.png")
        settings_image = Image.open("./images/settings-svgrepo-com.png")
        data_image = Image.open("./images/data-svgrepo-com.png")
        build_image = Image.open("./images/build-svgrepo-com.png")
        self.backtestImage = ctk.CTkImage(dark_image = chart_image, size=(50,50))
        self.realTimeImage = ctk.CTkImage(dark_image = dollar_image, size=(50,50))
        self.settingsImage = ctk.CTkImage(dark_image = settings_image, size=(50,50))
        self.dataImage = ctk.CTkImage(dark_image = data_image, size=(50,50))
        self.buildImage = ctk.CTkImage(dark_image = build_image, size=(50,50))
        
        #buttons
        button_backtest = ctk.CTkButton(self.sidebar, text="Backtest", font=self.navbar_font, compound=BOTTOM, command=self.load_historical, image=self.backtestImage, width=60,height=60)
        button_realtime = ctk.CTkButton(self.sidebar, text="Real-Time", font=self.navbar_font, compound=BOTTOM, command=self.load_realtime, image=self.realTimeImage, width=60,height=60)
        button_build = ctk.CTkButton(self.sidebar, text="Build Model", font=self.navbar_font, compound=BOTTOM, command=self.load_build_model, image=self.buildImage, width=60,height=60)
        button_data = ctk.CTkButton(self.sidebar, text="Process Data", font=self.navbar_font, compound=BOTTOM, command=self.load_process_data, image=self.dataImage, width=60,height=60)
        button_setting = ctk.CTkButton(self.sidebar, text="Settings", font=self.navbar_font, compound=BOTTOM, command=self.load_settings, image=self.settingsImage, width=60,height=60)
        
        #grid positioning
        button_backtest.grid(row=1, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_realtime.grid(row=2, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_build.grid(row=3, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_data.grid(row=4, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_setting.grid(row=5, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")

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
        #print(value)
        self.layer_widgets[index]['type'] = value
        #print(self.layer_widgets)

        #in relation to building the model
    def update_layer_param(self, event, index):
        entry_widget = event.widget
        current_text = entry_widget.get()  # This gets the current text from the entry
        #print(current_text)
        self.layer_widgets[index]['params'] = float(current_text)
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
                
                layer_type_combo = ctk.CTkComboBox(self.building_layers_frame, values=["LSTM", "Dense", "Dropout"])
                layer_type_combo.grid(row=i, column=1, padx=15, pady=15)

                layer_param_entry = ctk.CTkEntry(self.building_layers_frame)
                layer_param_entry.grid(row=i, column=2, padx=15, pady=15)
                
                self.layer_widgets.append({'type': layer_type_combo.get(), 'params': layer_param_entry})

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
            for info in layer_info:
                layer_type = info['type']
                if layer_type == 'LSTM':
                    model.add(LSTM(info['params'], return_sequences=True, input_shape=input_shape))
                elif layer_type == 'Dropout':
                    model.add(Dropout(info['params']))
                elif layer_type == 'Dense':
                    model.add(Dense(info['params']))

            return model
        
        def collect_model_details():
            layer_details = []
            #print(self.layer_widgets)
            for layer_info in self.layer_widgets:
                #print(layer_info['type'], layer_info['params'])
                layer_type = layer_info['type']
                layer_params = layer_info['params']
                layer_details.append({'type': layer_type, 'params': float(layer_params)})

            hyperparameters = {
                'learning_rate': learning_rate_entry.get(),
                'epochs': epochs_entry.get(),
                'batch_size': batch_size_entry.get(),
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
            model.save(hyper_params['model name'] + '.h5')  # Save the model

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
        parameters_title = ctk.CTkLabel(self.main_frame, text = "Hyperparameter Configuration", font = self.combo_box_font, text_color="#353535")
        parameters_title.pack(padx=15,pady=15, side=TOP, anchor = "w")
        # Dropdowns for activation functions and optimizer choices.
        select_parameters_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        select_parameters_frame.pack(pady=15, side=TOP, fill = X, padx = 20)
        
        # Entries for learning rate, epochs, batch size, and sequence length (specifically important for LSTM models).
        # Learning Rate
        learning_rate_label = ctk.CTkLabel(select_parameters_frame, text="Learning Rate:")
        learning_rate_label.grid(row=1, column=0, padx=15, pady=15)
        learning_rate_entry = ctk.CTkEntry(select_parameters_frame)
        learning_rate_entry.grid(row=1, column=1, padx=15, pady=15)
        # Epochs
        epochs_label = ctk.CTkLabel(select_parameters_frame, text="Epochs:")
        epochs_label.grid(row=2, column=0, padx=15, pady=15)
        epochs_entry = ctk.CTkEntry(select_parameters_frame)
        epochs_entry.grid(row=2, column=1, padx=15, pady=15)
        # Batch Size
        batch_size_label = ctk.CTkLabel(select_parameters_frame, text="Batch Size:")
        batch_size_label.grid(row=3, column=0, padx=15, pady=15)
        batch_size_entry = ctk.CTkEntry(select_parameters_frame)
        batch_size_entry.grid(row=3, column=1, padx=15, pady=15)
        # Sequence Length In
        sequence_length_in_label = ctk.CTkLabel(select_parameters_frame, text="Sequence Length In:")
        sequence_length_in_label.grid(row=4, column=0, padx=15, pady=15)
        sequence_length_in_entry = ctk.CTkEntry(select_parameters_frame)
        sequence_length_in_entry.grid(row=4, column=1, padx=15, pady=15)
        # Sequence Length Out
        sequence_length_out_label = ctk.CTkLabel(select_parameters_frame, text="Sequence Length Out:")
        sequence_length_out_label.grid(row=5, column=0, padx=15, pady=15)
        sequence_length_out_entry = ctk.CTkEntry(select_parameters_frame)
        sequence_length_out_entry.grid(row=5, column=1, padx=15, pady=15)
        # Optimiser
        optimiser_label = ctk.CTkLabel(select_parameters_frame, text="Optimiser")
        optimiser_label.grid(row=1, column=2, padx=15, pady=15)
        optimiser_entry = ctk.CTkEntry(select_parameters_frame)
        optimiser_entry.grid(row=1, column=3, padx=15, pady=15)
        # Loss
        loss_function_label = ctk.CTkLabel(select_parameters_frame, text="Loss Function")
        loss_function_label.grid(row=2, column=2, padx=15, pady=15)
        loss_function_entry = ctk.CTkEntry(select_parameters_frame)
        loss_function_entry.grid(row=2, column=3, padx=15, pady=15)

        # Submission: Button to confirm the setup and proceed to data preparation.
        model_build_button_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        model_build_button_frame.pack(pady=20, side=TOP, fill = X, padx = 20)

        #have a button that hits backtest, which calls the backtest function, which will return a html and we will embed that,
                                                                                                        #comamnd missing here
        model_build_button = ctk.CTkButton(model_build_button_frame, text="Build Model", font=self.button_font, command=lambda: build_and_save_model())
        model_build_button.grid(row=0, column=0, padx=15, pady=15)

    def load_process_data(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(self.main_frame, text = "process data")
        label.pack(pady=20)

        #Data Upload:
        # Button to upload dataset files.
        # Display of uploaded file paths to confirm the data is loaded.
        # Feature Engineering:
        # Checkboxes to select standard and optional features (e.g., price, volume, RSI, MACD).
        # Entries to specify parameters for computing technical indicators.
        # Data Scaling/Normalization:
        # Radio buttons to choose between scaling methods (e.g., Min-Max Scaling, Standard Scaling).
        # Sequence Preparation:
        # Entry for specifying sequence length, crucial for LSTM models.
        # Button to trigger the sequence creation process, using the sequence length and selected features.
        # Display of sample sequences to verify correct formation.

    def load_realtime(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(self.main_frame, text = "realtime area")
        label.pack(pady=20)

    def load_settings(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(self.main_frame, text = "settings area")
        label.pack(pady=20)

        # Combo box for theme selection
        self.theme_combobox = ctk.CTkComboBox(self.main_frame, values=["Dark", "Light"], command=self.change_theme)
        self.theme_combobox.set("Dark")  # Set the default value to match the initial theme
        self.theme_combobox.pack(pady=20)

    def change_theme(self, event=None):
        # Change the appearance mode based on the combo box selection
        selected_theme = self.theme_combobox.get()
        self._set_appearance_mode(selected_theme)

    
        


if __name__ == "__main__":
    app = MainWindow
    app.mainloop()