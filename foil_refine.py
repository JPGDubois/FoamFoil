"""
AIRFOIL REFINING FILE

PARAMETERS: "airfoil file path", number of points parameter

RETURNS: open airfoil points 2D array, closed airfoil points 2D array

For the airfoil file path, you can place the file in the desktop and use:
"Users\\[USERNAME]\\Desktop\\[AIRFOIL FILE.txt]"
(don't forget the quotation marks)

Number of points parameter DOES NOT define the exact number of points in
the returned arrays. The number of points will be at least LESS than
2.5 times the parameter. Recommended values: 200-300.

Use Selig format dat files from airfoiltools.com or export from XFLR5 saved
as .txt instead of .dat!!!

In this format, the files have a single point list that starts at x = 1
(TE), loops around the LE and goes back to x = 1.

The airfoils output by this file will probably not run through X-Foil analysis
optimally. If using for VLM, try refining them with the application's own
option for it, such as "refine globally" in XFLR5.
"""

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


class Airfoil:

        def __init__(self, foil):
                self.foil = foil

        def rotate(self,xarr,yarr,ox,oy,radsangle):
                """
                Rotate a list of points counterclockwise by a given angle around a given origin.

                The angle should be given in radians.
                """

                return [ox + math.cos(-angle) * (xarr[i] - ox) - math.sin(-angle) * (yarr[i] - oy) for i in range(len(xarr))], [oy + math.sin(-angle) * (xarr[i] - ox) + math.cos(-angle) * (yarr[i] - oy) for i in range(len(yarr))]

        def translate(self,xarr,yarr,xdist,ydist):
                """
                Translate by xdist and ydist
                """

                return [x + xdist for x in xarr], [x + ydist for x in yarr]

        def resize(self,xarr,yarr,chord):
                """
                Resize to chord with origin [0, 0]
                Use only for derotated airfoils with LE at [0,0]
                """

                self.chord_ratio = chord*2/(xarr[0] + xarr[-1])

                return [x*self.chord_ratio for x in xarr], [x*self.chord_ratio for x in yarr]

        def foil_refine(self, npoints):

                self.x = self.foil[:,0]
                self.y = self.foil[:,1]

                self.lex = []
                self.ley = []

                #Making LE spline to find accurate LE for normalization and derotation
                for i in range(len(self.x)):
                        if self.x[i] < 0.05:
                                self.lex.append(self.x[i])
                                self.ley.append(self.y[i])

                #Linear point spacing in y, more points near the LE
                self.new_ley = np.linspace(min(self.ley), max(self.ley), 200)
                self.new_lex = si.interpolate.interp1d(self.ley, self.lex, kind='cubic')(self.new_ley)

                self.tex = (self.x[0] + self.x[-1])/2
                self.tey = (self.y[0] + self.y[-1])/2

                self.c = []

                for i in range(len(self.new_lex)):
                self.c.append( np.sqrt( (self.new_lex[i] - self.tex)**2 + (self.new_ley[i] - self.tey)**2 ) )

                self.xle = self.new_lex[np.argmax(self.c)]
                self.yle = self.new_ley[np.argmax(self.c)]

                self.x, self.y = self.rotate( self.x, self.y, self.xle, self.yle, -np.arctan((self.tey-self.yle)/(self.tex-self.xle)) )

                self.x, self.y = self.translate( self.x, self.y, -self.xle, -self.yle )

                self.x, self.y = self.resize( self.x, self.y, 1 )

                for i in range(len(self.x)):
                        if self.x[i] < 0.05:
                                if self.y[i] < 0:
                                        self.indexsplit = i
                                        break

                if self.y[self.indexsplit-1] - self.y[self.indexsplit] < 0:
                        self.x = np.flip(self.x)
                        self.y = np.flip(self.y)

                        for i in range(len(self.x)):
                                if self.x[i] < 0.05:
                                        if self.y[i] < 0:
                                                self.indexsplit = i
                                                break

                self.xu = np.concatenate((self.x[:self.indexsplit],[0]), axis=None)
                self.yu = np.concatenate((self.y[:self.indexsplit],[0]), axis=None)

                self.xl = np.concatenate(([0],self.x[self.indexsplit:]), axis=None)
                self.yl = np.concatenate(([0],self.y[self.indexsplit:]), axis=None)

                del self.lex
                del self.ley
                del self.new_lex
                del self.new_ley
                del self.tex
                del self.tey
                del self.c
                del self.xle
                del self.yle

                if npoints % 2 != 0:
                        npoints += 1

                #Sine-weighted spacing for the x distribution of interpolated points, more points near TE and LE
                self.x_new = [x/2 + 0.5 for x in np.sin(np.linspace(-(math.pi/2), (math.pi/2), npoints/2))]

                self.yu_new = si.interpolate.Akima1DInterpolator(self.xu, self.yu)(self.x_new)
                self.yl_new = si.interpolate.Akima1DInterpolator(self.xl, self.yl)(self.x_new)

                self.x = np.concatenate( ( np.flip(self.x_new),self.x_new[1:] ), axis=None )
                self.y = np.concatenate( ( np.flip(self.yu_new),self.yl_new[1:] ), axis=None )

                self.foil = [self.x,self.y]

                del self.x_new
                del self.yu_new
                del self.yl_new
                del self.x
                del self.y

        def getfoil(self):
                return self.foil
