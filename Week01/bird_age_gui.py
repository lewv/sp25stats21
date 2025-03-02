#!/usr/bin/env python3

"""
bird_age_gui.py

This script prompts the user a bird's age and returns the human age equivalent
and also uses a gui

You may need to install tkinter if it is not part of your standard Python install.

Usage:
------
Run the script and follow the prompts:
$ python bird_age_gui.py
"""

import tkinter as tk
from tkinter import messagebox

# Define conversion functions
def bird_to_human_years(age):
    if age <= 0:
        return "Age must be a positive number."
    return age * 5

# Function to calculate human age based on pet type and input
def calculate_age():
    try:
        bird_age = float(age_entry.get())
        if bird_age <= 0:
            raise ValueError  # Manually raise a ValueError below
        
        human_age = bird_to_human_years(bird_age)
        
        messagebox.showinfo("Result", f"Your bird is about {human_age}")
    
    except ValueError:
        messagebox.showerror("Error", "Please input a valid age.")


# Create the GUI window
root = tk.Tk()
root.title("Pet Age Calculator")
root.geometry("300x250")

# Pet Type Dropdown
tk.Label(root, text="Select Pet Type:").pack(pady=5)
pet_type_var = tk.StringVar()
pet_type_var.set("Bird")  # Default option
pet_options = ["Bird"]
pet_menu = tk.OptionMenu(root, pet_type_var, *pet_options)
pet_menu.pack()

# Age Input
tk.Label(root, text="Enter Bird Age:").pack(pady=5)
age_entry = tk.Entry(root)
age_entry.pack()

# Calculate Button
calculate_btn = tk.Button(root, text="Calculate Human Age", command=calculate_age)
calculate_btn.pack(pady=10)

# Run the GUI
root.mainloop()
