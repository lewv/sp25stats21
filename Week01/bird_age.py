#!/usr/bin/env python3

"""
bird_age.py

This script prompts the user a bird's age and returns the human age equivalent

Usage:
------
Run the script and follow the prompts:
$ python bird_age.py
"""

def bird_to_human_years(bird_age):
    if bird_age <= 0:
        return "Age must be a positive number."
    return f"Your bird is {bird_age * 5} in human years"

bird_age = float(input("What the bird's age? "))
print(bird_to_human_years(bird_age))