"""
Builds 3D geometry of a two-airfoil section.

Use setters for geometrical paramaters.

root and tip attributes are the actual geometry (3D arrays containing the coordinates of the two foils).

Needs a main with a guy to call it. Needs a projecter that will project the geometry onto the planes that
the hotwire ends move along.
"""

import numpy as np
import scipy.interpolate as si
import math

#Class that creates a two-foil section from geometric parameters
class Section:
        def __init__(self,rootfoil,tipfoil):
                self.rootfoil = rootfoil
                self.tipfoil = tipfoil

                self.npoints = 150

                self.rightside = True
                self.chord = [1,1]
                self.span = [0,1]
                self.offset = [0,0]
                self.twist = [0,0]
                #self.sweep = None
                self.dihedral = [0,0]
                self.yoffset = 0
                self.r25 = [[0.25, 0, 0],[0.25, 0, 1]] #(reference points at 0.25c)
                self.o = [0.25,0.25] #rotation centre (chord fraction)

                self.root = Airfoil(self.rootfoil)
                self.tip = Airfoil(self.tipfoil)

                #self.root = None
                #self.tip = None

        def set_npoints(self, n):
                self.npoints = n
        def set_side(self, side):
                if side == 'r':
                        self.rightside = True
                if side == 'l':
                        self.rightside = False
                else:
                        raise ValueError("Side can only be 'r' or 'l'")
        def set_chord(self, a, b):
                self.chord = [a,b]
        def set_span(self, a, b):
                self.span = [a,b]
        def set_offset(self, a,b):
                self.offset = [a,b]
        def set_twist(self, a,b):
                self.offset = [a,b]
        def set_dihedral(self, a,b):
                self.dihedral = [a,b]
        def r25_yoffset(y):
                self.yoffset = y
                self.r25[:,1] = [x + y for x in self.r25[:,1]]
        def set_o(self, a,b):
                self.o = [a,b]

        def get_side(self):
                return self.rightside

        def get_r25(self):
                return self.r25

        def get_foils(self):
                return self.root.foil, self.tip.foil

        #builds the 3D geometry
        def build(self):
                #refines foils to npoints points
                self.oroot.foil_refine(self.npoints)
                self.otip.foil_refine(self.npoints)

                root = self.root.foil
                tip = self.tip.foil

                #rescales to chord
                root = Airfoil.resize(root[0], root[1], self.chord[0])
                tip = Airfoil.resize(tip[0], tip[1], self.chord[1])
                self.r25[:,0] = [0.25*self.chord[0], 0.25*self.chord[1]]

                #twist rotation
                root = Airfoil.rotate(root[0], root[1], self.o[0], 0, math.radians(self.twist[0]))
                tip = Airfoil.rotate(tip[0], tip[1], self.o[1], 0, math.radians(self.twist[1]))
                #add functionality for tracking r25 if o != 0.25

                #adds z-coordinate (span)
                root = np.vstack((root, zeros(len(root))))
                tip = np.vstack((tip, zeros(len(tip))))

                #dihedral rotation
                root[1], root[2] = Airfoil.rotate(root[1], root[2], 0, 0, math.radians(self.dihedral[0]))
                tip[1], tip[2] = Airfoil.rotate(tip[1], tip[2], 0, 0, math.radians(self.dihedral[1]))
                #add functionality for tracking r25 if o != 0.25

                #offset and span translation
                root[0], root[2] = Airfoil.translate(root[0], root[2], self.offset[0], self.span[0])
                tip[0], tip[2] = Airfoil.translate(tip[0], tip[2], self.offset[1], self.span[1])
                #add functionality for tracking r25 if o != 0.25

                #y-axis translation from dihedral and root y-offset
                root[1], root[2] = Airfoil.translate(root[1], root[2], self.yoffset, 0)
                tip[1], tip[2] = Airfoil.translate(tip[1], tip[2], np.sin(math.radians(self.dihedral[0]))*(self.span[1]-self.span[0]) + self.yoffset, 0)
                #add functionality for tracking r25 if o != 0.25

                self.root.foil = root
                self.tip.foil = tip

        class Airfoil:

                def __init__(self, foil):
                        self.ofoil = foil
                        self.foil = None

                def set_foil(self, foil):
                        self.ofoil = foil

                def get_foil(self):
                        return self.foil

                def rotate(xarr,yarr,ox,oy,radsangle):
                        """
                        Rotate a list of points counterclockwise by a given angle around a given origin.

                        The angle should be given in radians.
                        """

                        return [ox + math.cos(-angle) * (xarr[i] - ox) - math.sin(-angle) * (yarr[i] - oy) for i in range(len(xarr))], [oy + math.sin(-angle) * (xarr[i] - ox) + math.cos(-angle) * (yarr[i] - oy) for i in range(len(yarr))]

                def translate(xarr,yarr,xdist,ydist):
                        """
                        Translate by xdist and ydist
                        """

                        return [x + xdist for x in xarr], [x + ydist for x in yarr]

                def resize(xarr,yarr,chord):
                        """
                        Resize to chord with origin [0, 0]
                        Use only for derotated airfoils with LE at [0,0]
                        """

                        chord_ratio = chord*2/(xarr[0] + xarr[-1])

                        return [x*chord_ratio for x in xarr], [x*chord_ratio for x in yarr]

                def foil_refine(self, npoints):

                        x = self.ofoil[:,0]
                        y = self.ofoil[:,1]

                        lex = []
                        ley = []

                        #Takes small set of points near LE to find accurate LE
                        for i in range(len(x)):
                                if x[i] < 0.05:
                                        lex.append(x[i])
                                        ley.append(y[i])

                        #Sets up a linear point spacing and does spline interpolation of the LE curve
                        new_ley = np.linspace(min(ley), max(ley), 200)
                        new_lex = si.interpolate.interp1d(ley, lex, kind='cubic')(new_ley)

                        #finds TE
                        tex = (x[0] + x[-1])/2
                        tey = (y[0] + y[-1])/2

                        c = []

                        #finds distance between every point of interpolated LE section and TE
                        for i in range(len(new_lex)):
                                c.append( np.sqrt( (new_lex[i] - tex)**2 + (new_ley[i] - tey)**2 ) )

                        #finds most distant point from TE (a.k.a. most accurate LE)
                        xle = new_lex[np.argmax(c)]
                        yle = new_ley[np.argmax(c)]

                        #derotates
                        x, y = rotate( x, y, xle, yle, -np.arctan((tey-yle)/(tex-xle)) )

                        #translates so LE is at [0,0]
                        x, y = translate( x, y, -xle, -yle )

                        #normalizes so that TE is at [1,0]
                        x, y = resize( x, y, 1 )

                        #splits foil into upper and lower curves
                        for i in range(len(x)):
                                if x[i] < 0.05:
                                        if y[i] < 0:
                                                indexsplit = i
                                                break

                        #flips if the foil starts on the underside
                        if y[indexsplit-1] - y[indexsplit] < 0:
                                x = np.flip(x)
                                y = np.flip(y)

                                for i in range(len(x)):
                                        if x[i] < 0.05:
                                                if y[i] < 0:
                                                        indexsplit = i
                                                        break

                        #adds the LE to both curves
                        xu = np.concatenate((x[:indexsplit],[0]), axis=None)
                        yu = np.concatenate((y[:indexsplit],[0]), axis=None)

                        xl = np.concatenate(([0],x[indexsplit:]), axis=None)
                        yl = np.concatenate(([0],y[indexsplit:]), axis=None)

                        #makes sure npoints is even
                        if npoints % 2 != 0:
                                npoints += 1

                        #Sine-weighted spacing for the x distribution of interpolated points, more points near TE and LE
                        x_new = [x/2 + 0.5 for x in np.sin(np.linspace(-(math.pi/2), (math.pi/2), npoints/2))]

                        #akima interpolation is less susceptible to oscillations near endpoints
                        yu_new = si.interpolate.Akima1DInterpolator(xu, yu)(x_new)
                        yl_new = si.interpolate.Akima1DInterpolator(xl, yl)(x_new)

                        #joins the two curves (excluding the LE from the lower curve to avoid duplicate) to form a single foil
                        x = np.concatenate( ( np.flip(x_new),x_new[1:] ), axis=None )
                        y = np.concatenate( ( np.flip(yu_new),yl_new[1:] ), axis=None )

                        self.foil = [x,y]

#Sec argument has to be an instance of Section
class Coordinates:
	def __init__(self, Sec):

	self.Sec = Sec

		#sets the root and tip foils to the correct side
		if Section.get_side() == True:
			self.l, self.r = Sec.get_foils()
			self.rightside = True
		else:
			self.r, self.l = Sec.get_foils()
			self.rightside = False

	self.machine_width = 300

	#Machine coordinates
	self.lc = None
	self.rc = None

	def set_machine_width(self, width_mm):
		self.machine_width = width_mm

	def get_machine_width():
		return self.machine_width

	def get_coordinates():
		return self.lc, self.rc

	def rotate(xarr,yarr,ox,oy,radsangle):
		"""
		Rotate a list of points counterclockwise by a given angle around a given origin.

		The angle should be given in radians.
		"""

		return [ox + math.cos(-angle) * (xarr[i] - ox) - math.sin(-angle) * (yarr[i] - oy) for i in range(len(xarr))], [oy + math.sin(-angle) * (xarr[i] - ox) + math.cos(-angle) * (yarr[i] - oy) for i in range(len(yarr))]

	def translate(arr,dist):
		"""
		Translate by xdist and ydist
		"""

		return [x + dist for x in arr]

	def generate(self):

		#translates so the middle point of the 0.25c line is at x=0, z=0
		z_center = (self.Sec.get_r25[0, 2] + self.Sec.get_r25[1, 2])/2
		l_z = translate( self.l[2], -z_center)
		r_z = translate( self.r[2], -z_center)

		x_center = (self.Sec.get_r25[0, 0] + self.Sec.get_r25[1, 0])/2
		l_x = translate( self.l[0], -x_center)
		r_x = translate( self.r[0], -x_center)

		#calculates sweep angle
		if self.rightside == True:
			x_sweep = np.arctan( (self.Sec.get_r25[1, 0] - self.Sec.get_r25[0, 0])/((self.Sec.get_r25[1, 2] - self.Sec.get_r25[0, 2])) )
		else:
			x_sweep = -np.arctan( (self.Sec.get_r25[1, 0] - self.Sec.get_r25[0, 0])/((self.Sec.get_r25[1, 2] - self.Sec.get_r25[0, 2])) )

		#derotates sweep
		l_x, l_z = rotate(self.l[0], self.l[2], 0, 0, -x_sweep)
		r_x, r_z = rotate(self.r[0], self.r[2], 0, 0, -x_sweep)

		l_y = self.l[1]
		r_y = self.r[1]

		#projects points to machine-planes in x and then y
		lc_x = [l_x[i] + ((self.machine_width/2) - l_z[i]) * (l_x[i] - r_x[i])/(l_z[i] - r_z[i]) for i in range(len(l_x))]
		rc_x = [r_x[i] + ((self.machine_width/2) - r_z[i]) * (r_x[i] - l_x[i])/(r_z[i] - l_z[i]) for i in range(len(r_x))]

		lc_y = [l_y[i] + ((self.machine_width/2) - l_z[i]) * (l_y[i] - r_y[i])/(l_z[i] - r_z[i]) for i in range(len(l_y))]
		rc_y = [r_y[i] + ((self.machine_width/2) - r_z[i]) * (r_y[i] - l_y[i])/(r_z[i] - l_z[i]) for i in range(len(l_y))]

		self.lc = np.vstack((lc_x, lc_y))
		self.rc = np.vstack((rc_x, rc_y))
