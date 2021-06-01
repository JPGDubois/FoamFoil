"""
Converts list of coords to gcode


G0 = fast movement, goes to location as fast as possible
G1 = cutting at specified feedrate (In combination with F)
F = feed rate in mm/min
M5 = turn off hot wire
M3 = turn on hot wire (in combination with S)
S = percentage of voltage to hot wire
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
    self.foam_size = [100, 50, 300] #chord, hight, span
    self.Coords = Coords

    self.lu, self.ll, self.ru, self.rl = None
    #pts = [[11,12,13,14],[21,22,23,24],[31,32,33,34]]

    def rotate(self,x,y,ox,oy,angle):
            """
            Rotate a list of points counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """

            return ox + math.cos(-angle) * (x - ox) - math.sin(-angle) * (y - oy), oy + math.sin(-angle) * (x - ox) + math.cos(-angle) * (y - oy)



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

    #steps through all the coordiantes for the cut top side (work in proggress)
    def cut_upper(self):
        gcode = ['G1']
        lstart = self.lu[]

        #Modify feedrate depending on point density, only one side is considered since they are equivalent.
        dl = [np.sqrt((self.lu[0,i+1] - self.lu[0,i])**2+(self.lu[1,i+1] - self.lu[1,i])**2) for i in range(len(self.lu[0])-1)]


        #list ranging from 0 to 1
        factor = [(i-min(dl))/(max(dl)-min(dl)) for i in dl]

        fl = [i*(self.F[1]-self.F[0])+self.F[0] for i in factor]
        ml = [i*(self.F[0]-self.F[1])+self.F[1] for i in factor]

        F = np.hstack([[fl[0]],fl]) #list of feedrates
        M = np.hstack([[ml[0]],ml]) #list of cutting temperatures

        for i in range(len(self.lu[0])): #make the gcode

            line = str(self.ax1+str(self.lu[0,i])+ ' ' +self.ax2+str(self.lu[1,i])+ ' ' + self.ax3+str(self.ru[0,i])+' '+ self.ax4+str(self.ru[1,i])+ ' F' + str(F[i]) + ' M3 S' + str(M[i]))
            gcode.append(line)


        gcode.append('M5')
        return gcode



    file = start(name)
    file = rapid(file, pts)
    file = cut(file, pts, F, M)
    file = end(file)
    print(file)
