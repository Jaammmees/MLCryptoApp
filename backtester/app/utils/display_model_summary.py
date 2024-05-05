import io
from contextlib import redirect_stdout
import customtkinter as ctk
from tensorflow.keras.models import load_model

def display_model_summary(model_path, display_frame, font):
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
    header, rows, footer = parse_summary(summary_lines)

    # Clear previous widgets from the frame
    for widget in display_frame.winfo_children():
        widget.destroy()

    # Display the new model summary in the frame
    for line in header + rows + footer:
        label = ctk.CTkLabel(master=display_frame, text=line, text_color='#FFFFFF', font=font)
        label.pack(pady=2, padx=10, anchor="w")

    display_frame.pack(padx=15, pady=15, expand=True, fill='both')

def parse_summary(lines):
    header = lines[0:4]  # Assuming first 4 lines include the headers and separators
    rows = lines[4:-5]  # The model layers are listed after the header and before the total params
    footer = lines[-5:]  # Total params and other details
    return header, rows, footer