# Daniel Karkhut
# Handeling User Input

import tkinter as tk
from tkinter import filedialog

class Input:

    def __init__(self):
        self.blur = 1 # default 1
        self.dilate = 3 # default 3
        self.cellSize = 60 # default 60
        
    def getBlurAndDilate(self):
        while True:
            try:
                # Get input from user
                self.blur = int(input("Please enter the degree of Gaussian Blur [default 1]: "))
                self.dilate = int(input("Please enter the degree of Contour Dilation [default 3]: "))
                self.cellSize = int(input("Please enter the max length and width of cells [default 100]: ")) 
                if (self.cellSize % 2) != 0:
                    self.cellSize = self.cellSize + 1
            except ValueError:
                print("Please input an integer.")
                # Return to the start of the loop
                continue
            else:
                # blur and dilate was successfully parsed!
                break
        return self.blur, self.dilate, self.cellSize

    def getFolderLocation(self, string):
        tk.Tk().withdraw() # remove unnecessary TK window
        print(string)
        folder = filedialog.askdirectory()
        return folder 