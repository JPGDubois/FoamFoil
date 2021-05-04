import os
name = "Users\kievb\Desktop\Projects\Icarus\AG 26.dat"
#Making the list

def get(name):
    """
    if os.path.exists(str(name)):
        # Reading a file
        with open(str(name),'r') as g:# Open file for reading
            pts = []
            for i in g.readlines():
                line = i.strip('\n').split(' ')
                ln = []
                for j in range(len(line)):
                    line[j] = line[j].lstrip()
                    if len(line[j]) != 0:
                        ln.append(line[j])
                    else:
                        continue
                #print(ln)
                pts.append(ln)
        #print(pts)
        #getting dat header
        header = ' '.join(pts[0])
        del pts[0]

    else:
        exit()

    return(header,pts)
    """
import pandas as pd
import numpy as np
import scipy.interpolate as si
import math
from operator import itemgetter
import matplotlib.pyplot as plt


foil = "C:\\"+str(name)

df = pd.read_csv(foil,sep='\s+',skiprows=(1),header=(0))

brutefoil = df.values

import foil_refine

wing = foil_refine.Section(brutefoil, brutefoil)
wing.build()

l, r = wing.get_foils()

plt.plot(l[0], l[1])
plt.show()
