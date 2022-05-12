import sys
import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import yaml
import transform as t

'''
Makes a 2d plot from all airfoils in a list of Airfoil objects.
'''
def visualize_foil(airfoils):
    fig = plt.figure('Airfoils', figsize=(12, 6))
    ax = fig.add_subplot(111)
    for obj in airfoils:
        name = obj.get_name()
        ofoil = obj.get_ofoil()
        x = ofoil[:, 0]
        y = ofoil[:, 2]
        ax.plot(x, y, label = name, marker='.')
        ax.legend()
    ax.set_title('Airfoils')
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, 0.2])
    plt.show()

def visualize_section(Sec, Prof):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure('Section', figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    data1 = Sec.get_foils()
    data2 = Prof.get_profiles()
    qcpoint = Sec.get_qcpoint()
    rootName = Sec.root.get_name()
    tipName = Sec.tip.get_name()
    xRoot1 = data1[0][:,0]
    yRoot1 = data1[0][:,1]
    zRoot1 = data1[0][:,2]
    xTip1 = data1[1][:,0]
    yTip1 = data1[1][:,1]
    zTip1 = data1[1][:,2]
    xqcRoot = qcpoint[0][0]
    yqcRoot = qcpoint[0][1]
    zqcRoot = qcpoint[0][2]
    xqcTip = qcpoint[1][0]
    yqcTip = qcpoint[1][1]
    zqcTip = qcpoint[1][2]
    xRoot2 = data2[0][:,0]
    yRoot2 = data2[0][:,1]
    zRoot2 = data2[0][:,2]
    xTip2 = data2[1][:,0]
    yTip2 = data2[1][:,1]
    zTip2 = data2[1][:,2]
    ax.plot(xRoot1, yRoot1, zRoot1, alpha = 1, label = rootName, marker='.')
    ax.scatter(xqcRoot, yqcRoot, zqcRoot, alpha = 1)
    ax.plot(xTip1, yTip1, zTip1, alpha = 1, label = tipName, marker='.')
    ax.scatter(xqcTip, yqcTip, zqcTip, alpha = 1)
    ax.plot(xRoot2, yRoot2, zRoot2, alpha = 1, marker='.')
    ax.plot(xTip2, yTip2, zTip2, alpha = 1, marker='.')
    ax.legend()
    plt.show()

def export_dat(Prof):
    data = Prof.get_profiles()
    xRoot = data[0][:,0]
    zRoot = data[0][:,2]
    root = np.array([xRoot, zRoot]).T
    xTip = data[1][:,0]
    zTip = data[1][:,2]
    tip = np.array([xTip, zTip]).T
    np.savetxt('cutRoot.txt', root, delimiter='  ')
    np.savetxt('cutTip.txt', tip, delimiter='  ')

def read_xwimp(directory):
    root = tk.Tk()
    root.withdraw()
    fileTypes = (
        ('xwimp files', '*.xwimp'),
        ('All files', '*.*')
        )
    filePath = fd.askopenfilename(
        title='Open wing geometry',
        initialdir=directory,
        filetypes=fileTypes
        )
    if filePath == '':
        sys.exit('no file selected')
    with open(filePath, 'r') as f:
        wingName = f.readline()
    sections = []
    data = pd.read_csv(filePath, delim_whitespace = True, header = None, skiprows = 1).to_numpy()
    sections = []
    for i in range(len(data)-1):
        sections.append(t.Section(f'Section_{i}'))
        sections[i].set_root(data[i])
        sections[i].set_tip(data[i+1])
        air = sections[i].set_foils(directory)
        visualize_foil(air)
    return sections

def get_project_folder(default = '/'):
    root = tk.Tk()
    root.withdraw()
    directory = fd.askdirectory(
        title='Select file path',
        initialdir=default,
        )
    if directory == '':
        sys.exit('no folder selected')
    return directory

def set_preset(prof, i = 0):
    with open('preset.yaml') as file:
        try:
            presets = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

        key = list(presets.keys())[0]
        prof.set_cutting_voltage(presets[key]['cuttingVoltage'])
        prof.set_cutting_feed(presets[key]['cuttingFeed'])
        prof.set_kerf(presets[key]['rootKerf'], presets[key]['tipKerf'])
        print(f'Preset {presets[key]["name"]} selected')

dir = get_project_folder()
Sec = read_xwimp(dir)
for i in Sec:
    i.set_npoints(2000)
    i.build()
    #i.align_le()
    i.locate_section()
    prof = t.Profile(i)
    set_preset(prof, 0)

    # Set distance between axis.
    prof.set_yspan(1250)

    prof.cutting_planes()
    visualize_section(i, prof)
    #export_dat(prof)
    prof.paths()

    # Export to file
    prof.coords_to_gcode(dir, mirror = False)

    prof.coords_to_gcode(dir, mirror = True)
    print(f'Section done')
