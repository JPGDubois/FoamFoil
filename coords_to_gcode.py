"""
Converts list of coords to gcode
"""
import numpy as np
import math
import foil_refine
import pandas as pd



class Gcode:
    def __init__(self, Coords):

    #pts = np.random.rand(200,4)

    self.name = 'O0000'
    self.ax1 = 'X'
    self.ax2 = 'Y'
    self.ax3 = 'U'
    self.ax4 = 'Z'
    self.M = [300,500]
    self.F = [100,250]
    self.rapid_plane_dist = 15
    self.foam_size = [100, 50, 300]
    self.Coords = Coords

    self.lu, self.ll, self.ru, self.rl = None
    #pts = [[11,12,13,14],[21,22,23,24],[31,32,33,34]]



    def splitter(self):
        #blahblah something something (Write later)
        arr = self.Coords.get_coordinates()

        self.lu = [ arr[0, 0, :-(-len(arr[0,0])//2)+1, arr[0, 1, :-(-len(arr[0,1])//2)+1 ]
        self.ll = [ arr[0, 0, -(-len(arr[0,0])//2):, arr[0, 1, -(-len(arr[0,1])//2): ]
        self.ru = [ arr[1, 0, :-(-len(arr[1,0])//2)+1, arr[1, 1, :-(-len(arr[1,1])//2)+1 ]
        self.rl = [ arr[1, 0, -(-len(arr[1,0])//2):, arr[1, 1, -(-len(arr[1,1])//2): ]

    #starts a list containing all commands
    def start(self):
        return ['%','('+self.name+') G21 G90 M5']

    #steps through all the coordinates of the rapid movement
    #not happy with this approach but should work. it needs the last known coordinate, moves straight up to the ceiling height
    #does a horizontal movement above the target position and goes straight down to the target point.
    def rapid(self, ceiling, start, target):
        gcode = ['G0']

        gcode.append(self.ax1+str(start[0])+self.ax2+str(ceiling)+self.ax3+str(start[2])+self.ax4+str(ceiling))
        gcode.append(self.ax1+str(target[0])+self.ax2+str(ceiling)+self.ax3+str(target[2])+self.ax4+str(ceiling))
        gcode.append(self.ax1+str(target[0])+ self.ax2+str(target[1])+self.ax3+str(target[2])+self.ax4+str(target[3]))
        return gcode

    #terminates the list
    def end(self):
        return ['M5','%']

    def cut(self):
        return ('G1' + 'M3 S'+ str(self.M[0]))

    #steps through all the coordiantes for the cut (work in proggress)
    def cut_upper(self):
        gcode = []
        lstart = lu[]
        dl = [np.sqrt((lu[0,i+1] - lu[0,i])**2+(lu[1,i+1] - lu[1,i])**2) for i in range(len(lu[0])-1)]
        dr = [np.sqrt((ru[0,i+1] - ru[0,i])**2+(ru[1,i+1] - ru[1,i])**2) for i in range(len(ru[0])-1)]
        for i in range(len(pts)):

            line = str(self.ax1+str(lu[0,i])+ ' ' +self.ax2+str(lu[1,i])+ ' ' + self.ax3+str(ru[0,i])+' '+ self.ax4+str(ru[1,i])+ ' F' + str(F) + ' M3 S' + str(M))
            gcode.append(line)
        gcode.append('M5')
        return gcode


    #returns a factor that corresponds to the taper ratio at that specific point
    #The difference in distance travelled in both planes scales with k.
    def tapercorrect(this, pts):
        k = []
        for i in range(len(this.pts)-1):
            d1 = np.sqrt((this.pts[i][0]-this.pts[i+1][0])**2 + (this.pts[i][1]-this.pts[i+1][1])**2)
            d2 = np.sqrt((this.pts[i][2]-this.pts[i+1][2])**2 + (this.pts[i][3]-this.pts[i+1][3])**2)
            dd = np.abs(d1 - d2)
            k.append(dd)
        return k




    file = start(name)
    file = rapid(file, pts)
    file = cut(file, pts, F, M)
    file = end(file)
    print(file)
