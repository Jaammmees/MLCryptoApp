#library imports
import customtkinter as ctk
from tkinter import *
from PIL import Image
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input # type: ignore
from utils.display_model_summary import display_model_summary
from utils.data_handling import load_data_file, load_data_file_and_modify, process_and_save_data
from utils.model_management import load_model_file, load_model_preview, load_model_file_return_shapes, start_training
from utils.sequence_processing import generate_sequence
from utils.scaler_management import load_scaler_file, upload_scaler_prompt

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

        #makes historical default
        self.load_historical()

    def setup_sidebar(self):
        """
        Sets up the sidebar with navigation buttons each linked to a specific functionality within the application.

        This method loads images for buttons, creates button widgets, and configures their grid placement in the sidebar.
        """

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

        #sidebar positioning
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
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        path_container = {}

        #title
        backtest_title = ctk.CTkLabel(self.main_frame, text="Backtesting", font=self.title_font, text_color="#353535")
        backtest_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        #loading frame
        loader_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        loader_frame.pack(pady=5, side=TOP, fill = X, padx = 20)

        #need to be able to load a machine model, and a scalar, choose the data we want to backtest it on LOCALLY or just a TICKER that we can grab
        #the data has to be in the form INDEX == Timestep, Open, High, Low, Close, Volume as of now
        #buttons and text for loading model, scaler, and data
        model_button = ctk.CTkButton(loader_frame, text="Load Model (.h5)", font=self.button_font, command=lambda : load_model_file(path_container, model_indicator))
        model_indicator = ctk.CTkLabel(loader_frame, text="No Model Selected")
        scaler_button = ctk.CTkButton(loader_frame, text="Load Scaler (.pkl)", font=self.button_font, command=lambda : load_scaler_file(path_container, scaler_indicator))
        scaler_indicator = ctk.CTkLabel(loader_frame, text="No Scaler Model Selected")
        data_button = ctk.CTkButton(loader_frame, text="Load Data (Excel/Parquet)", font=self.button_font, command=lambda : load_data_file(path_container, data_indicator))
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

    #------------------------------------------------------------------------------------------------------------------------------------------------

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
            display_model_summary(model_path, self.model_preview_frame, self.button_font)

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

        # Button to load a model
        model_button = ctk.CTkButton(load_files_frame, text="Load Model (.h5)", font=self.button_font, command= lambda : load_model_preview(path_container, model_indicator, model_preview_frame, display_model_summary, self.button_font))
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
                                command=lambda: load_data_file_and_modify(path_container, self.columns,data_indicator, columns_frame, self.column_list_frame, delete_column))
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
        choose_scaler_combo = ctk.CTkComboBox(normalisation_frame, values=['MinMaxScaler', 'StandardScalar', 'Upload Scaler'], 
                                      command=lambda value: upload_scaler_prompt(value, normalisation_frame, path_container, self.button_font))
        choose_scaler_combo.grid(row=0,column=1,padx=15,pady=15)

        # Sequence Preparation:
        # intake the model shape for the sequence length, crucial for LSTM models.
        data_process_button_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        data_process_button_frame.pack(pady=15,side=TOP,fill=X,padx=20)

        data_process_button = ctk.CTkButton(data_process_button_frame, font=self.button_font, text="Process Data & Save", command=lambda: process_and_save_data(path_container, self.columns, data_preview_frame, choose_scaler_combo))
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
        3. Display dynamic feedback on the model and data status, and provide a preview area for generated sequences.
        """

        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        process_data_title = ctk.CTkLabel(self.main_frame, text="Model Training", font=self.title_font, text_color="#353535")
        process_data_title.pack(pady=20,padx=25, side=TOP, anchor = "w")

        path_container = {}
        self.model_input_shape = None
        self.model_output_shape = None

        def update_shapes():
            input_shape, output_shape = load_model_file_return_shapes(path_container, model_indicator, model_shape)
            if input_shape and output_shape:
                self.model_input_shape = input_shape
                self.model_output_shape = output_shape

        load_files_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        load_files_frame.pack(pady=15,side=TOP,fill=X,padx=20)

        #model stuff
        model_indicator = ctk.CTkLabel(load_files_frame, text="No Model Selected")
        model_indicator.grid(row = 1, column = 0, padx=15, pady=15)
        model_shape = ctk.CTkLabel(load_files_frame, text="")
        model_shape.grid(row=2, column = 0, padx=15,pady=15)
        model_shape.grid_remove()
        model_button = ctk.CTkButton(load_files_frame, text="Load Model (.h5)", font=self.button_font, command=lambda: update_shapes())
        model_button.grid(row=0, column = 0, padx=15, pady=15)

        #data stuff
        data_indicator = ctk.CTkLabel(load_files_frame, text="No Data Selected")
        data_indicator.grid(row = 1, column = 2, padx=15, pady=15)
        data_button = ctk.CTkButton(load_files_frame, text="Load Data (Excel/Parquet)", font=self.button_font, command= lambda: load_data_file(path_container, data_indicator))
        data_button.grid(row=0, column = 2, padx=15, pady=15)
        
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
        sequence_apply_button = ctk.CTkButton(sequence_generation_frame, text="Generate Sequence", command=lambda: generate_sequence(self.model_input_shape, sequence_length_entry.get(), sequence_preview_text, path_container))
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
        train_model_button = ctk.CTkButton(hyperparameter_frame, text="Start Training", command= lambda : start_training(path_container, sequence_preview_text, sequence_length_entry.get(), learning_rate_entry.get(), epochs_entry.get()))
        train_model_button.grid(row=3, column=1, padx=15, pady=15)

    def load_realtime(self):
        #wipe previous frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(self.main_frame, text = "realtime area")
        label.pack(pady=20)
