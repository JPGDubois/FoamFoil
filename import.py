"""
Imports only dat files with:
- 1st line header
- coordinates looping around (not first LE -> TE top, LE -> TE bottom)

"""
import os
name = "mh45.dat"

import pandas as pd
import numpy as np
import scipy.interpolate as si
import math
from operator import itemgetter
import matplotlib.pyplot as plt

foil = str(name)         # Relative path
# foil = "C:\\"+str(name)       # Absolute path

df = pd.read_csv(foil,sep='\s+',skiprows=(1),header=(0))

brutefoil = df.values

import foil_refine

wing = foil_refine.Section(brutefoil, brutefoil)
wing.build()

l, r = wing.get_foils()
print(l)
plt.plot(l[0], l[1])
plt.show()
