import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog as fd

# prompt_foil prompts the user to select the used airfoils. These airfoil data points are
# stored as a numpy array in a dictionary with as key the first line of the file.

def prompt_foil():
    root = tk.Tk()
    root.withdraw()
    fileTypes = (
        ('dat files', '*.dat'),
        ('text files', '*.txt'),
        ('comma separated values', '*.csv'),
        ('All files', '*.*')
        )
    filePaths = fd.askopenfilenames(
        title='Open airfoils',
        initialdir='/',
        filetypes=fileTypes
        )

    foilDict = {}
    for path in filePaths:
        with open(path, 'r') as f:
            foilName = f.readline().strip()
        points = pd.read_csv(path, delim_whitespace = True, header = None, skiprows = 1).astype(float)
        x = points.to_numpy()[:, 0]
        y = np.zeros(len(x))
        z = points.to_numpy()[:, 1]
        foilDict[foilName] = np.array([x, y, z]).T
    return foilDict


def visualize_2d(dict):
    print(dict)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    for key in dict:
        print(dict[key])
        x = dict[key][:, 0]
        y = dict[key][:, 2]
        ax.scatter(x, y, label = key)
    ax.set_title('Airfoils')
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, 0.2])
    plt.show()


    def visualize_3d(data):
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data = np.random.random(size=(3, 3, 3))
        ax.scatter(data[:,0],data[:,1],data[:,2], c='red', alpha=1)
        plt.show()

visualize_2d(prompt_foil())
