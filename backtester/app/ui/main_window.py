import customtkinter as ctk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
from trading.historical import run_backtest_and_save_report
import tkinterweb
import threading


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
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("#EDEDED", "#4B4B4B"))
        #customise
        self.main_frame.pack(side="right", expand=True, fill="both")
        
        #fonts
        self.navbar_font = ctk.CTkFont(family = "Helvetica", size = 12, weight = "bold")
        self.title_font = ctk.CTkFont(family = "Helvetica", size = 40, weight = "bold")
        self.button_font = ctk.CTkFont(family = "Helvetica", size = 15, weight = "bold")

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
        self.backtestImage = ctk.CTkImage(dark_image = chart_image, size=(50,50))
        self.realTimeImage = ctk.CTkImage(dark_image = dollar_image, size=(50,50))
        self.settingsImage = ctk.CTkImage(dark_image = settings_image, size=(50,50))
        
        #buttons
        button_backtest = ctk.CTkButton(self.sidebar, text="Backtest", font=self.navbar_font, compound=BOTTOM, command=self.load_historical, image=self.backtestImage, width=60,height=60)
        button_realtime = ctk.CTkButton(self.sidebar, text="Real-Time", font=self.navbar_font, compound=BOTTOM, command=self.load_realtime, image=self.realTimeImage, width=60,height=60)
        button_setting = ctk.CTkButton(self.sidebar, text="Settings", font=self.navbar_font, compound=BOTTOM, command=self.load_settings, image=self.settingsImage, width=60,height=60)
        
        #grid positioning
        button_backtest.grid(row=1, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_realtime.grid(row=2, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")
        button_setting.grid(row=3, column = 0, pady=20, padx=30, ipady=8, ipadx=8, sticky="nsew")

        self.sidebar.grid_columnconfigure(0, weight=1)  # Make column 0 take up all available space
        self.sidebar.grid_rowconfigure(0, weight=1)  # Spacer row at the top
        self.sidebar.grid_rowconfigure(1, weight=0)  # Actual button row
        self.sidebar.grid_rowconfigure(2, weight=0)  # Actual button row
        self.sidebar.grid_rowconfigure(3, weight=0)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(4, weight=1)  # Spacer row at the bottom
        self.sidebar.grid_rowconfigure(5, weight=1)  # Spacer row at the bottom


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

        def backtest_button_callback():
            # Load data and create strategy instance here, or ensure they are accessible
            data = EURUSD.copy()
            
            # Run backtest in a separate thread to avoid freezing the GUI
            thread = threading.Thread(target=lambda: run_backtest_and_update_gui(data))
            thread.start()

        def run_backtest_and_update_gui(data):
            results, report_path = run_backtest_and_save_report(data)
            display_backtest_results(backtest_result_frame, report_path, results)
        
        def display_backtest_results(master, html_file, results):
            # Clear previous results if necessary
            for widget in master.winfo_children():
                widget.destroy()

            # Frame for the web view
            web_frame = ctk.CTkFrame(master)
            web_frame.pack(fill='both', expand=True)

            # Display the HTML report
            html_view = tkinterweb.HtmlFrame(web_frame, horizontal_scrollbar="auto")
            html_view.load_file(html_file)  # Ensure html_file is the correct, full path
            html_view.pack(fill='both', expand=True)

            # Optionally display the results in a text widget or labels
            results_text = ctk.CTkLabel(master, text=str(results))
            results_text.pack(pady=20)


        #-----------------------------------------------------------------------
        
        #backtesting frame
        backtest_button_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        backtest_button_frame.pack(pady=20, side=TOP, fill = X, padx = 20)

        #have a button that hits backtest, which calls the backtest function, which will return a html and we will embed that,

        backtest_button = ctk.CTkButton(backtest_button_frame, text="Run Backtest", font=self.button_font, command=backtest_button_callback)
        backtest_button.grid(row=0, column=0, padx=15, pady=15)


        #backtest results frame

        backtest_result_frame = ctk.CTkFrame(self.main_frame, corner_radius= 10)
        backtest_result_frame.pack(pady=10, side=TOP, fill = BOTH, padx = 20)

        # Create a frame for the web view
        web_frame = ctk.CTkFrame(backtest_result_frame)
        web_frame.pack(fill='both', expand=True)

        # Setup the HTML viewer within the frame
        html_file_path = "C:/Users/LimJ/Documents/GitHub/MLCrypto/backtester/app/backtest_report.html"
        self.html_view = tkinterweb.HtmlFrame(web_frame, horizontal_scrollbar="auto")
        self.html_view.load_file(html_file_path)
        self.html_view.pack(fill='both', expand=True)




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