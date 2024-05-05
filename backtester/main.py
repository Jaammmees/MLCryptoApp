from ui.main_window import MainWindow

import customtkinter as ctk

def main():
    ctk.set_appearance_mode("System")  # Could be 'Dark' or 'Light' or 'System'
    ctk.set_default_color_theme("blue")  # Set the default color theme

    app = MainWindow()
    app.mainloop()

if __name__ == '__main__':
    main()