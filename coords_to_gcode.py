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
    self.cut_direction = 'te'
    self.Coords = Coords
    self.gcode = []

    self.lu, self.ll, self.ru, self.rl = None
    #pts = [[11,12,13,14],[21,22,23,24],[31,32,33,34]]

    def set_cut_direction(self, direction):
        if direction == 'le':
            self.cut_direction = 'le'
        else if direction == 'te':
            self.cut_direction = 'te'
        else if direction == 'te-le':
            self.cut_direction = 'te-le'
        else:
            raise Exception("cut direction must be 'le', 'te' or 'te-le'")

    def get_cut_direction(self):
        return self.cut_direction

    def get_gcode():
        return self.gcode

    def rotate(self,x,y,ox,oy,angle):
            """
            Rotate a list of points counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """

            return ox + math.cos(-angle) * (x - ox) - math.sin(-angle) * (y - oy), oy + math.sin(-angle) * (x - ox) + math.cos(-angle) * (y - oy)



    def splitter(self):
        #blahblah something something (Write later)
        arr = self.Coords.get_coordinates()

        self.lu = [ arr[0, 0, :(len(arr[0,0])//2)+1, arr[0, 1, :(len(arr[0,1])//2)+1 ]
        self.ll = [ arr[0, 0, (len(arr[0,0])//2):, arr[0, 1, (len(arr[0,1])//2): ]
        self.ru = [ arr[1, 0, :(len(arr[1,0])//2)+1, arr[1, 1, :(len(arr[1,1])//2)+1 ]
        self.rl = [ arr[1, 0, (len(arr[1,0])//2):, arr[1, 1, (len(arr[1,1])//2): ]

    #starts a list containing all commands
    def start(self):
        return ['%','('+self.name+') G21 G90 M5']

    #steps through all the coordinates of the rapid movement
    #not happy with this approach but should work. it needs the last known coordinate, moves straight up to the ceiling height
    #does a horizontal movement above the target position and goes straight down to the target point.

    #moves to rapid planing, zone can be "le", "te" or "ceil" for the front, back and top. le and te will result in horizontal motion
    #returns list of form [gcode line, final position]
    def to_rapid(self, position, zone = "ceil"):
        if zone == "ceil":
            return ['G0' + ' ' + self.ax2 + str(self.foam_size[1]+self.rapid_plane_dist) + ' ' +  self.ax4 + str(self.foam_size[1]+self.rapid_plane_dist), [position[0], self.foam_size[1]+self.rapid_plane_dist, position[2], self.foam_size[1]+self.rapid_plane_dist] ]

        else if zone == "le":
            return ['G0' + ' ' + self.ax1 + str(-self.rapid_plane_dist) + ' ' +  self.ax3 + str(-self.rapid_plane_dist), [-self.rapid_plane_dist, position[1], -self.rapid_plane_dist, position[3]] ]

        else if zone == "te":
            return ['G0' + ' ' + self.ax1 + str(self.foam_size[0]+self.rapid_plane_dist) + ' ' +  self.ax3 + str(self.foam_size[0]+self.rapid_plane_dist), [self.foam_size[0]+self.rapid_plane_dist, position[1], self.foam_size[0]+self.rapid_plane_dist, position[3]] ]

        else:
            raise ValueError("zone must be 'le', 'te' or 'ceil'")

    #can only be used after using to_rapid first
    #moves along rapid plane (perimeter) to the desired position target = [left coord, right coord]
    #If zone == 'ceil', the target must be in chord-coordinates, otherwise target must be in height-coordinates

    def rapid(self, position, target, zone = "ceil"):

        if not (position[0] = -self.rapid_plane_dist and position[2] = -self.rapid_plane_dist) or (position[0] = self.foam_size[0] + self.rapid_plane_dist and position[2] = self.foam_size[0] + self.rapid_plane_dist) or (position[1] = self.foam_size[1] + self.rapid_plane_dist and position[3] = self.foam_size[1] + self.rapid_plane_dist):

            raise Exception("Machine not moved to rapid plane before using rapid(). Use to_rapid() before using rapid()")

        gcode = []

        if zone == "ceil":

            if target < -self.rapid_plane_dist or target > self.foam_size[0] + self.rapid_plane_dist:
                raise ValueError("target not within rapid plane")

            if position[1] != self.foam_size[1]+self.rapid_plane_dist or position[3] != self.foam_size[1]+self.rapid_plane_dist:

                gcode.append(self.ax2 + str(self.foam_size[1]+self.rapid_plane_dist) + ' ' +  self.ax4 + str(self.foam_size[1]+self.rapid_plane_dist)

            gcode.append(self.ax1 + str(target[0]) + ' ' + self.ax3 + str(target[1]))
            return [gcode, [target[0], self.foam_size[1]+self.rapid_plane_dist, target[1], self.foam_size[1]+self.rapid_plane_dist]]


        else if zone == "le":

            if target < 0 or target > self.foam_size[1] + self.rapid_plane_dist:
                raise ValueError("target not within rapid plane")

            if position[0] != -self.rapid_plane_dist or position[2] != -self.rapid_plane_dist:

                if position[1] != self.foam_size[1]+self.rapid_plane_dist or position[3] != self.foam_size[1]+self.rapid_plane_dist:
                    gcode.append(self.ax2 + str(self.foam_size[1]+self.rapid_plane_dist) + ' ' +  self.ax4 + str(self.foam_size[1]+self.rapid_plane_dist)

                gcode.append(self.ax1 + str(-self.rapid_plane_dist) + ' ' +  self.ax3 + str(-self.rapid_plane_dist)

            gcode.append(self.ax2 + str(target[0]) + ' ' +  self.ax4 + str(target[1])

            return [gcode, [-self.rapid_plane_dist, target[0], -self.rapid_plane_dist, target[1]]]


        else if zone == "te":

            if target < 0 or target > self.foam_size[1] + self.rapid_plane_dist:
                raise ValueError("target not within rapid plane")

            if position[0] != self.foam_size[0] + self.rapid_plane_dist or position[2] != self.foam_size[0] + self.rapid_plane_dist:

                if position[1] != self.foam_size[1]+self.rapid_plane_dist or position[3] != self.foam_size[1]+self.rapid_plane_dist:
                    gcode.append(self.ax2 + str(self.foam_size[1]+self.rapid_plane_dist) + ' ' +  self.ax4 + str(self.foam_size[1]+self.rapid_plane_dist)

                gcode.append(self.ax1 + str(self.foam_size[0] + self.rapid_plane_dist) + ' ' +  self.ax3 + str(self.foam_size[0] + self.rapid_plane_dist)

            gcode.append(self.ax2 + str(target[0]) + ' ' +  self.ax4 + str(target[1])

            return [gcode, [-self.rapid_plane_dist, target[0], -self.rapid_plane_dist, target[1]]]

        else:
            raise ValueError("zone must be 'le', 'te' or 'ceil'")

    def cut(self):
        return ('G1' + 'M3 S'+ str(self.M[0]))

    #steps through all the coordiantes for the cut top side (work in proggress)
    def cut_upper(self):
        gcode = []

        #Modify feedrate depending on point density, only one side is considered since they are equivalent.
        dl = [np.sqrt((self.lu[0,i+1] - self.lu[0,i])**2+(self.lu[1,i+1] - self.lu[1,i])**2) for i in range(len(lu[0])-1)]

        fl = [((dl[i]-min(dl))/(max(dl)-min(dl)*(self.F[1]-self.F[0]))+self.F[0]) for i in range(len(self.lu[0])-1)]

        F = np.hstack([[fl[0]],fl]) #list of feedrates

        for i in range(len(self.lu)): #make the gcode

            line = str(self.ax1+str(self.lu[0,i])+ ' ' +self.ax2+str(self.lu[1,i])+ ' ' + self.ax3+str(self.ru[0,i])+' '+ self.ax4+str(self.ru[1,i])+ ' F' + str(F[i]) + ' M3 S' + str(self.M))
            gcode.append(line)


        gcode.append('M5')
        return gcode, [self.lu[-1], self.ru[-1]]

    def cut_lower(self):
        gcode = []

        #Modify feedrate depending on point density, only one side is considered since they are equivalent.
        dl = [np.sqrt((self.ll[0,i+1] - self.ll[0,i])**2+(self.ll[1,i+1] - self.ll[1,i])**2) for i in range(len(lu[0])-1)]

        fl = [((dl[i]-min(dl))/(max(dl)-min(dl)*(self.F[1]-self.F[0]))+self.F[0]) for i in range(len(self.ll[0])-1)]

        F = np.hstack([[fl[0]],fl]) #list of feedrates

        for i in range(len(self.ll)): #make the gcode

            line = str(self.ax1+str(self.ll[0,i])+ ' ' +self.ax2+str(self.ll[1,i])+ ' ' + self.ax3+str(self.rl[0,i])+' '+ self.ax4+str(self.rl[1,i])+ ' F' + str(F[i]) + ' M3 S' + str(self.M))
            gcode.append(line)


        gcode.append('M5')
        return gcode, [self.ll[-1], self.rl[-1]]

    #builds the gcode
    def build(self):

        splitter(self)

        gcode = [start(self)]

        if self.cut_direction == 'te':

            ll = np.flip(ll)
            rl = np.flip(rl)

            line, pos = to_rapid(self, [0, self.foam_size[1], 0, self.foam_size[1]])

            gcode.append(line)

            line, pos = rapid(self, pos, [self.lu[1][0], self.ru[1][0]], zone = 'te')

            gcode = np.hstack([gcode, [line], cut(self)])

            lines, pos = cut_upper(self)

            gcode = np.hstack([gcode, lines])

            line, pos = to_rapid(self, pos, zone = 'le')

            gcode.append(line)

            gcode.append('M5')

            line, pos = rapid(self, pos, [self.lu[1][-1], self.ru[1][-1]], zone = 'te')

            gcode = np.hstack([gcode, [line], cut(self)])

            lines, pos = cut_lower(self)

            gcode = np.hstack([gcode, lines])

            line, pos = to_rapid(self, pos, zone = 'le')

            gcode.append('M5')

            gcode = np.hstack([ gcode, [line], rapid(self, pos, [-self.rapid_plane_dist, -self.rapid_plane_dist])[0] ])

            gcode.append('M2')

            self.gcode = gcode
