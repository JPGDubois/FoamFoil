"""
Converts list of coords to gcode
"""
import numpy as np

pts = np.random.rand(200,4)

name = 'test'
ax1 = 'X'
ax2 = 'Y'
ax3 = 'U'
ax4 = 'Z'
M = 300
F = 200
#pts = [[11,12,13,14],[21,22,23,24],[31,32,33,34]]

#starts a list containing all commands
def start(name):
    gcode = ['('+name+')','G90','M5']
    return gcode

#steps through all the coordinates of the rapid movement
def rapid(gcode, pts):
    for i in range(len(pts)):
        line = str('G0 '+ax1+str(pts[i][0])+ ' ' +ax2+str(pts[i][1])+ ' ' + ax3+str(pts[i][2])+' '+ ax4+str(pts[i][3]))
        gcode.append(line)
    return gcode

#terminates the list
def end(gcode):
    gcode.append('M5')
    return gcode

#steps through all the coordiantes for the cut
def cut(gcode, pts, F, M):
    gcode.append('M3 S'+ str(M))
    for i in range(len(pts)):
        line = str('G1 '+ax1+str(pts[i][0])+ ' ' +ax2+str(pts[i][1])+ ' ' + ax3+str(pts[i][2])+' '+ ax4+str(pts[i][3])+ ' F' + str(F))
        gcode.append(line)
    gcode.append('M5')
    return gcode
"""
def cut_Var(gcode, pts, F, M):
    gcode.append('M3 S'+ str(M))
    for i in range(len(pts)):
        line = str('G1 '+ax1+str(pts[i][0])+ ' ' +ax2+str(pts[i][1])+ ' ' + ax3+str(pts[i][2])+' '+ ax4+str(pts[i][3])+ ' F' + str(F))
        gcode.append(line)
    gcode.append('M5')
    return gcode

def distance(pts):
    k = []
    for i in range(len(pts)-1):
        d1 = np.sqrt((pts[i][0]-pts[i+1][0])**2 + (pts[i][1]-pts[i+1][1])**2)
        d2 = np.sqrt((pts[i][2]-pts[i+1][2])**2 + (pts[i][3]-pts[i+1][3])**2)


        k.append(d1)
    return k
"""

file = start(name)
file = rapid(file, pts)
file = cut(file, pts, F, M)
file = end(file)
print(file)
