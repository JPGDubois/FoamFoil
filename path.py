import numpy as np
import foil_refine
import pandas as pd

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
"""

class wing_path:
    def __init__(self, name, coords, roof):
        # Parameters for the path generation
        self.name = str(name)   # gcode name
        self.coords = coords    # input lists with foil points
        self.roof = 0           # height at which rapids are performed
        self.leadin_len = 0     # lenght of lead in at TE
        #self.leadout_len = 0    # lenght of lead out at LE
        self.o = [0, 0, 0, 0]   # gcode origen

        # Splitted airfoil coordinates
        self.lu = []    # [[x, y], ...] left upper
        self.ll = []    # [x, y], ...] left lower
        self.ru = []    # [z, u], ...] right upper
        self.rl = []    # [z, u], ...] right lower

        # Parameters for gcode generation
        self.gcode = [] # gcode commands
        self.ax1 = 'X'  # axis name first axis
        self.ax2 = 'Y'  # axis name second axis
        self.ax3 = 'U'  # axis name third axis
        self.ax4 = 'Z'  # axis name fourth axis
        self.M = [300, 500] # cutting temperature range
        self.F = [100, 250] # cutting feed rate range

    def split():
        # Split coords into different section; left upper, left lower, right upper and right cut_lower
        arr = self.Coords.get_coordinates()

        self.lu = [ arr[0, 0, :-(-len(arr[0, 0])//2)+1, arr[0, 1, :-(-len(arr[0, 1])//2)+1 ]
        self.ll = [ arr[0, 0, -(-len(arr[0, 0])//2):, arr[0, 1, -(-len(arr[0, 1])//2): ]
        self.ru = [ arr[1, 0, :-(-len(arr[1, 0])//2)+1, arr[1, 1, :-(-len(arr[1, 1])//2)+1 ]
        self.rl = [ arr[1, 0, -(-len(arr[1, 0])//2):, arr[1, 1, -(-len(arr[1, 1])//2): ]
        return

    def rapid_upper(self):
        # Moves is a rectangular path to the start of the upper cut,
        # the height of the cut is determined by roof.
        start = self.o
        end = [self.lu[0]+self.leadin_len, self.lu[1], self.ru[0]+self.leadin_len, self.ru[1]]
        path = [[start[0], self.roof, start[2], self.roof], [end[0], self.roof, end[2], self.roof], end]
        return path

    def rapid_lower(self):
        # Moves is a rectangular path to the start of the lower cut,
        # the height of the cut is determined by roof.
        start = self.o
        end = [self.ll[0]+self.leadin_len, self.ll[1], self.rl[0]+self.leadin_len, self.rl[1]]
        path = [[start[0], self.roof, start[2], self.roof], [end[0], self.roof, end[2], self.roof], end]
        return path

    def cut_upper(self):
        # Do a straight lead in, trace the airfoil upper path
        # and do the lead out tangential to the airfoil LE.
        path = []
        for i in range(len(self.lu)):
            path.append(self.lu[i] + self.ru[i])
        path.append(self.rotate(path[-1], 5, -np.pi/2))
        path.append([self.o[0], path[-1]])
        return path

    def cut_lower(self):
        # Do a straight lead in, trace the airfoil lower path
        # and do the lead out tangential to the airfoil LE.
        path = []
        for i in range(len(self.ll)):
            path.append(self.ll[i] + self.rl[i])
        path.append(self.rotate(path[-1], 5, np.pi/2))
        path.append([self.o[0], path[-1]])
        return path

    def rotate(self, start, R, endangle):
        # Make a list of points on a circular path used for the tangential leadout.
        oxl = start[0] - R
        oyl = start[1]
        oxr = start[2] - R
        oyr = start[3]
        angle = 0
        path = []
        while angle >= endangle:
            pathl = [oxl + np.cos(-angle) * (start[0] - oxl) - np.sin(-angle) * (start[1] - oyl), oyl + np.sin(-angle) * (start[0] - oxl) + np.cos(-angle) * (start[1] - oyl)]
            pathr = [oxr + np.cos(-angle) * (start[2] - oxr) - np.sin(-angle) * (start[3] - oyr), oyr + np.sin(-angle) * (start[2] - oxr) + np.cos(-angle) * (start[3] - oyr)]
            path.append(pathl + pathr)
            angle -= np.pi/40
        return path

    def home(self):
        # Return the home location
        return self.o

    def G0(self, path):
        # Convert the rapid path to machine code
        self.gcode.append('G0')

        for i in path:
            line = str(self.ax1+str(i[0])+ ' ' +self.ax2+str(i[1])+ ' ' + self.ax3+str(i[2])+' '+ self.ax4+str(i[3]))
            self.gcode.append(line)
        return self.gcode

    def G1(self, path, F, M):
        # Convert the cutting path to machine code with feed rate modulation
        #
        # Determine the feed rate and cutting temperature based on the point density

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

    def start(self):
        # Return the starting lines of machine code
        self.gcode.extend(['%','('+self.name+') G21 G90 M5'])
        return self.gcode

    def end(self):
        # Return the ending lines of gcode
        self.gcode.extend(['M5','%'])
        return self.gcode

    def export(self):
        # Write a txt file containing the machine code
        df = pd.DataFrame(salary)
        df.to_csv(str(self.name) + '.csv', index=False, header=False)
