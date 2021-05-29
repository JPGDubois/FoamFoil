import numpy as np
import math
import foil_refine
"""
Current approach:
    cut top and bottom surface from TE to LE.
        Rapid 1:
        -> go up to roof on x=0 vertical
        -> traverse to beginning TE cut over horizontal at y = roof

        Cut 1:
        -> compute lead in TE
        -> go down to beginning lead of TE
        -> do lead in TE
        -> cut upper surface
        -> cut lead out in LE

        Rapid 2: (identical to rapid 1 but roof can be different)
        -> go up to roof on x=0 vertical
        -> traverse to beginning TE cut over horizontal at y = roof

        Cut 2: (identical to cut 1 but geometry is different)
        -> compute lead in TE
        -> go down to beginning lead of TE
        -> do lead in TE
        -> cut upper surface
        -> cut lead out in LE
Output different lists for each step,
each starting with a header containing the name and type(cut/rapid)
"""


class wing_path:
    def __init__(self, name, coords, roof):
        self.name = str(name)
        self.coords = coords
        self.roof = 0
        self.leadin_len = 0
        self.leadout_len = 0
        self.o = [0, 0, 0, 0]

        self.lu = []    # [[x, y], ...]
        self.ll = []    # [x, y]
        self.ru = []    # [z, u]
        self.rl = []    # [z, u]

        self.gcode = []
        self.ax1 = 'X'
        self.ax2 = 'Y'
        self.ax3 = 'U'
        self.ax4 = 'Z'
        self.M = [300,500]
        self.F = [100,250]

    def split():
        # do stuf to get self.lu,ll,ru,rl
        arr = self.Coords.get_coordinates()

        self.lu = [ arr[0, 0, :-(-len(arr[0, 0])//2)+1, arr[0, 1, :-(-len(arr[0, 1])//2)+1 ]
        self.ll = [ arr[0, 0, -(-len(arr[0, 0])//2):, arr[0, 1, -(-len(arr[0, 1])//2): ]
        self.ru = [ arr[1, 0, :-(-len(arr[1, 0])//2)+1, arr[1, 1, :-(-len(arr[1, 1])//2)+1 ]
        self.rl = [ arr[1, 0, -(-len(arr[1, 0])//2):, arr[1, 1, -(-len(arr[1, 1])//2): ]
        return()

    def rapid_upper(self):    # start, end [x, y, z, u]
        start = self.o
        end = [self.lu[0]+self.leadin_len, self.lu[1], self.ru[0]+self.leadin_len, self.ru[1]]
        path = [[start[0], self.roof, start[2], self.roof], [end[0], self.roof, end[2], self.roof], end]
        return path

    def rapid_lower(self):    # start, end [x, y, z, u]
        start = self.o
        end = [self.ll[0]+self.leadin_len, self.ll[1], self.rl[0]+self.leadin_len, self.rl[1]]
        path = [[start[0], self.roof, start[2], self.roof], [end[0], self.roof, end[2], self.roof], end]
        return path

    def cut_upper(self):
        path = []
        for i in range(len(self.lu)):
            path.append(self.lu[i] + self.ru[i])
        path.append(self.rotate(path[-1], 5, -math.pi/2))
        path.append([self.o[0], path[-1]])
        return path

    def cut_lower(self):
        path = []
        for i in range(len(self.ll)):
            path.append(self.ll[i] + self.rl[i])
        path.append(self.rotate(path[-1], 5, math.pi/2))
        path.append([self.o[0], path[-1]])
        return path

    def rotate(self, start, R, endangle):
        oxl = start[0] - R
        oyl = start[1]
        oxr = start[2] - R
        oyr = start[3]
        angle = 0
        path = []
        while angle >= endangle:
            pathl = [oxl + math.cos(-angle) * (start[0] - oxl) - math.sin(-angle) * (start[1] - oyl), oyl + math.sin(-angle) * (start[0] - oxl) + math.cos(-angle) * (start[1] - oyl)]
            pathr = [oxr + math.cos(-angle) * (start[2] - oxr) - math.sin(-angle) * (start[3] - oyr), oyr + math.sin(-angle) * (start[2] - oxr) + math.cos(-angle) * (start[3] - oyr)]
            path.append(pathl + pathr)
            angle -= math.pi/40
        return path

    def home(self):
        return self.o

    def G0(self, path):
        self.gcode.append('G0')
        for i in path:
            line = str(self.ax1+str(i[0])+ ' ' +self.ax2+str(i[1])+ ' ' + self.ax3+str(i[2])+' '+ self.ax4+str(i[3]))
            self.gcode.append(line)
        return self.gcode

    def G1(self, path, F, M):

        # Modify feedrate depending on point density, only one side is considered since they are equivalent.
        dl = [np.sqrt((path[i+1, 0] - path[i, 0])**2+(path[i+1, 1] - path[i, 1])**2) for i in range(len(path)-1)]

        # list ranging from 0 to 1
        factor = [(i-min(dl))/(max(dl)-min(dl)) for i in dl]

        fl = [i*(self.F[1]-self.F[0])+self.F[0] for i in factor]
        ml = [i*(self.F[0]-self.F[1])+self.F[1] for i in factor]

        F = np.hstack([[fl[0]], fl])  # list of feedrates
        M = np.hstack([[ml[0]], ml])  # list of cutting temperatures

        self.gcode.append('G1')
        for i in range(len(path)):
            line = str(self.ax1+str(path[i, 0])+' ' +self.ax2 + str(path[i, 1])+ ' ' +self.ax3+str(path[i, 2])+' '+self.ax4+str(path[i, 3])+' F' + str(F[i]) + ' M3 S' + str(M[i]))
            self.gcode.append(line)
        return self.gcode

    def Start(self):
        self.gcode.append('%','('+self.name+') G21 G90 M5')
        return self.gcode
        
    def end(self):
        self.gcode.append('M5','%')
        return self.gcode
