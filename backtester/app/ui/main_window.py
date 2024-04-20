import customtkinter as ctk
from tkinter import *
from PIL import Image, ImageTk

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
        
    
        self.setup_sidebar()

        #self.load_historical()

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
        button_backtest = ctk.CTkButton(self.sidebar, text="Backtest", compound=BOTTOM, command=self.load_historical, image=self.backtestImage, width=60,height=60)
        button_realtime = ctk.CTkButton(self.sidebar, text="Real-Time", compound=BOTTOM, command=self.load_realtime, image=self.realTimeImage, width=60,height=60)
        button_setting = ctk.CTkButton(self.sidebar, text="Settings", compound=BOTTOM, command=self.load_settings, image=self.settingsImage, width=60,height=60)
        
        #grid positioning
        button_backtest.grid(row=1, column = 0, pady=20, padx=30, ipady=5, ipadx=8, sticky="nsew")
        button_realtime.grid(row=2, column = 0, pady=20, padx=30, ipady=5, ipadx=8, sticky="nsew")
        button_setting.grid(row=3, column = 0, pady=20, padx=30, ipady=5, ipadx=8, sticky="nsew")

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

        label = ctk.CTkLabel(self.main_frame, text = "historical area")
        label.pack(pady=20)

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