import os
import tkinter as tk
import re
import threading
import main_adrs

def update_label():
    while True:
        value = main_adrs.measurement
        result_label.config(text=str(value))
        window.update()

def create_gui():
    window = tk.Tk()
    window.geometry("400x400")
    window.title("Analog Display Recognition System")

    # Font & Size
    window.configure(bg="#F0F0F0")
    font_title = ("Arial", 16, "bold")
    font_label = ("Arial", 12)
    font_entry = ("Arial", 11)

    title_label = tk.Label(window, text="Analog Display Recognizer", font=font_title, bg="#F0F0F0")
    title_label.pack(pady=20)

    # Labels
    value1 = tk.Label(window, text="Maximum", font=font_label, bg="#F0F0F0")
    value1.place(x=70, y=100)

    value2 = tk.Label(window, text="Minimum", font=font_label, bg="#F0F0F0")
    value2.place(x=70, y=140)

    unit_label = tk.Label(window, text="Unit", font=font_label, bg="#F0F0F0")
    unit_label.place(x=70, y=180)

    result_label = tk.Label(window, text="Result", font=font_label, bg="#F0F0F0")
    result_label.place(x=70, y=300)

    maximum = tk.Entry(window, font=font_entry, width=20)
    minimum = tk.Entry(window, font=font_entry, width=20)
    unit = tk.Entry(window, font=font_entry, width=20)

    maximum.place(x=150, y=100)
    minimum.place(x=150, y=140)
    unit.place(x=150, y=180)

    # Output Label
    output_label = tk.Label(window, font=font_label, bg="#F0F0F0")
    output_label.place(x=150, y=320)

    return window, result_label

if __name__ == "__main__":
    window, result_label = create_gui()

    # Start the thread to update the label
    update_thread = threading.Thread(target=update_label)
    update_thread.start()

    window.mainloop()
