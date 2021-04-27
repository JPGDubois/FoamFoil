import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate as si
import math
from operator import itemgetter

airfoilpath = setsmthplsss

foil = "C:\\"+str(airfoilpath)

df = pd.read_csv(foil,sep='\s+',skiprows=(1),header=(0))

foil = df.values
