import numpy as np
from scipy import interpolate as si
from scipy.linalg import expm, norm
from scipy.stats import linregress
import math
from copy import deepcopy
import os
import pandas as pd
import matplotlib.pyplot as plt

# Definition of rotation axis.
xAxis = np.array([1, 0, 0])
yAxis = np.array([0, 1, 0])
zAxis = np.array([0, 0, 1])

'''
The Airfoil class contains all operation functions that can be performed on
the airfoil coordinates. It also contains get_ functions to request certain
certain properties of the airfoil.
'''
class Airfoil:

    def __init__(self, foil, name = None):
        self.name = name    # Name of the airfoil.
        self.ofoil = foil   # Original foil datapoints (3d numpy array).
        self.foil = foil    # Foil datapoints (3d numpy array).
        self.qcpoint = np.array([0.25, 0., 0.]) # Quarter chord point (rotation center).
        self.lepoint = np.array([0., 0., 0.])   # Leading edge point.

    '''
    get_ functions return the requested property of the class.
    '''
    def get_name(self):
        return self.name

    def get_qcpoint(self):
        return self.qcpoint

    def get_lepoint(self):
        return self.lepoint

    def get_ofoil(self):
        return self.ofoil

    def get_foil(self):
        return self.foil

    def get_name(self):
        return self.name

    def set_qcpoint(self, qc):
        self.qcpoint = np.array([qc, 0., 0.])

    def update_ofoil(self):
        self.ofoil = deepcopy(self.foil)

    '''
    The scale, rotate and translate functions modify the foil coordinates
    described by the transformations. Note that the order of the transformations
    is important. First scale, then rotate and lastly translate.
    '''

    # Scales foil with a value of factor so that the chord has the desired dimension.
    # Scaling happens with the leading edge as center.
    def scale(self, chord):
        factor = chord*2/(self.foil[0, 0] + self.foil[-1, 0])
        self.foil =  factor * (self.foil - self.lepoint) + self.lepoint
        self.qcpoint = factor * (self.qcpoint - self.lepoint) + self.lepoint

    # axis is a unit vector of the axis along which the rotation is performed.
    # theta is the angle measured counterclockwise.
    def rotate(self, axis, theta):
        if math.isnan(theta):
            return
        rotationMatrix = expm(np.cross(np.eye(3), axis/norm(axis)*theta))
        self.qcpoint = np.dot(rotationMatrix, self.qcpoint.T).T
        self.lepoint = np.dot(rotationMatrix, self.lepoint.T).T
        self.foil = np.dot(rotationMatrix, self.foil.T).T

    # axis is a unit vector of the axis along which the translation is performed.
    # dist is the distance of the translation.
    def translate(self, axis, dist):
        translationVector = axis * dist
        self.foil += translationVector
        self.lepoint += translationVector
        self.qcpoint += translationVector

    '''
    The refine function (Originally written by Eduardo J. and modified) derotates
    and normalizes the foil and makes the point distribution uniform (equal amounts
    of points on top as on lower side with a density that varies sinusoidally).
    '''
    def refine(self, npoints):
        # Interpolate leading edge curve to get an accurate estimation of the lepoint.
        u = self.foil[:, 0]
        v = self.foil[:, 2]

        leu= []
        lev = []

        # Takes small set of points near LE to find accurate LE.
        for i in range(len(u)):
            if u[i] < 0.005:
                leu.append(u[i])
                lev.append(v[i])

        # Make sure all points are unique
        le = np.column_stack((leu, lev))
        le = le[np.unique(le[:,1], return_index=True)[1]]

        # Sets up a linear point spacing and does spline interpolation of the LE curve.
        newLev = np.linspace(min(lev), max(lev), 200)
        newLeu = si.interpolate.interp1d(le[:,1], le[:,0], kind='cubic')(newLev)

        # Finds TE by taking the average of first and last point.
        teu = (u[0] + u[-1])/2
        tev = (v[0] + v[-1])/2

        # Finds distance between every point of interpolated LE section and TE.
        c = []
        for i in range(len(newLeu)):
            c.append( np.sqrt( (newLeu[i] - teu)**2 + (newLev[i] - tev)**2 ) )

        # Finds most distant point from TE (a.k.a. most accurate LE).
        ule = newLeu[np.argmax(c)]
        vle = newLev[np.argmax(c)]

        # Find the angle of the chord line
        angle = np.arctan2(tev-vle, teu-ule)

        # Derotates the airfoil.
        self.rotate(yAxis, angle)

        # Translates so LE is at [0,0].
        self.translate(np.array([-ule, 0, -vle]), 1)

        # Normalizes so that TE is at [1,0].
        self.scale(1)

        # Get foil coords.
        u = self.foil[:, 0]
        v = self.foil[:, 2]

        # Normalise the trailing edge
        c = []
        for i in range(len(u)):
            if u[i] > 1.0:
                c.append(i)
        u = np.delete(u, c)
        v = np.delete(v, c)

        u[0] = 1.0
        u[-1] = 1.0
        v[0] = 0.0
        v[-1] = 0.0

        # Split airfoil at [0,0] (the LE).
        indexSplit = np.argmin(u)

        # Flip airfoil if it starts at the underside.
        if v[indexSplit-1] - v[indexSplit] < 0:
            u = np.flip(u)
            v = np.flip(v)

        # Adds the LE to both curves.
        ut = np.concatenate((u[:indexSplit],[0]), axis=None)
        vt = np.concatenate((v[:indexSplit],[0]), axis=None)

        ul = np.concatenate(([0],u[indexSplit:]), axis=None)
        vl = np.concatenate(([0],v[indexSplit:]), axis=None)

        # Flip top profiles such that all list of coordinates go from LE to TE.
        ut = np.flip(ut)
        vt = np.flip(vt)

        # Make sure the u values are stricktly increasing and no duplicates.
        top = np.column_stack((ut, vt))
        lower = np.column_stack((ul, vl))

        top = top[np.unique(top[:,0], return_index=True)[1]]
        lower = lower[np.unique(lower[:,0], return_index=True)[1]]

        # Makes sure npoints is even.
        if npoints % 2 != 0:
                npoints += 1

        # Sine-weighted spacing for the x distribution of interpolated points.
        # More points near TE and LE.
        uNew = [u/2 + 0.5 for u in np.sin(np.linspace(-(math.pi/2), (math.pi/2), int(npoints/2)))]

        # Akima interpolation is less susceptible to oscillations near endpoints.
        vtNew = si.Akima1DInterpolator(top[:,0], top[:,1])(uNew)
        vlNew = si.Akima1DInterpolator(lower[:,0], lower[:,1])(uNew)

        # Joins the two curves (excluding the LE from the lower curve to avoid duplicate) to form a single foil.
        u = np.concatenate( ( np.flip(uNew),uNew[1:] ), axis=None )
        v = np.zeros(npoints-1)
        w = np.concatenate( ( np.flip(vtNew),vlNew[1:] ), axis=None )

        self.foil = np.array([u, v, w]).T
        self.qcpoint = np.array([0.25, 0., 0.])
        self.lepoint = np.array([0., 0., 0.])

'''
The section class is responsible to house all the properties related to a wing
section. A wing section is the lofted body between two aifoil profiles. This class
is independent of machine properties
'''
class Section:

    def __init__(self, name):

        self.name = name

        self.npoints = 200  # Number of points on airfoil surface.
        self.unit = 1000    # Multiplier for imported values.

        self.chord = [1, 1]
        self.span = [0, 1]
        self.sweep = [0, 0] # Sweep offset in unit
        self.twist = [0, 0] # Twist angle in radians
        self.dihedral = [0, 0] # Dihedral angle in radians

        self.root = None    # Is an instance of the Airfoil class
        self.tip = None    # Is an instance of the Airfoil class

        self.rootName = None
        self.tipName = None

        self.zOffsetRoot = 7
        self.yOffsetRoot = 100
        self.xOffsetRoot = 10

    '''
    The set_ functions are used to configure the properties of the Section
    '''
    def set_root(self, xwimpLine):
        self.span[0] = self.unit*xwimpLine[0]
        self.chord[0] = self.unit*xwimpLine[1]
        self.sweep[0] = self.unit*xwimpLine[2]
        self.dihedral[0] = math.radians(xwimpLine[3])
        self.twist[0] = math.radians(xwimpLine[4])
        self.rootName = xwimpLine[9]

    def set_tip(self, xwimpLine):
        self.span[1] = self.unit*xwimpLine[0]
        self.chord[1] = self.unit*xwimpLine[1]
        self.sweep[1] = self.unit*xwimpLine[2]
        self.dihedral[1] = math.radians(xwimpLine[3])
        self.twist[1] = math.radians(xwimpLine[4])
        self.tipName = xwimpLine[9]

    def set_npoints(self, n):
        self.npoints = n

    def set_foils(self, filePath):
        for fileName in os.listdir(filePath):
            path = os.path.join(filePath, fileName)
            with open(path, 'r') as f:
                foilName = f.readline().strip().replace(' ', '/_/')

            if foilName == self.rootName:
                points = pd.read_csv(path, delim_whitespace = True, header = None, skiprows = 1).astype(float)
                x = points.to_numpy()[:, 0]
                y = np.zeros(len(x))
                z = points.to_numpy()[:, 1]
                self.root = Airfoil(np.array([x, y, z]).T, foilName)

            if foilName == self.tipName:
                points = pd.read_csv(path, delim_whitespace = True, header = None, skiprows = 1).astype(float)
                x = points.to_numpy()[:, 0]
                y = np.zeros(len(x))
                z = points.to_numpy()[:, 1]
                self.tip = Airfoil(np.array([x, y, z]).T, foilName)
        self.root.refine(self.npoints)
        self.tip.refine(self.npoints)
        self.root.update_ofoil()
        self.tip.update_ofoil()
        return [self.root, self.tip]

    def set_chord(self, a, b):
        self.chord = [a, b]

    def set_span(self, a, b):
        self.span = [a, b]

    def set_sweep(self, a, b):
        self.sweep = [a, b]

    def set_twist(self, a, b):
        self.twist = [a, b]

    def set_dihedral(self, a, b):
        self.dihedral = [a, b]

    def set_qcpoint(self, a, b):
        self.root.set_qcpoint(a)
        self.tip.set_qcpoint(b)

    def set_offset(self, a, b, c):
        self.zOffsetRoot = a
        self.yOffsetRoot = b
        self.xOffsetRoot = c

    def get_chord(self):
        return self.chord

    def get_span(self):
        return self.span

    def get_sweep(self):
        return self.sweep

    def get_twist(self):
        return self.twist

    def get_dihedral(self):
        return self.dihedral

    def get_foils(self):
        return self.root.foil, self.tip.foil

    def get_qcpoint(self):
        return self.root.get_qcpoint(), self.tip.get_qcpoint()

    '''
    The build function takes the two airfoils and moves them to the correct positions
    specified by the user to form the required wing.
    '''
    def build(self):
        # Refining the profiles.
        self.root.refine(self.npoints)
        self.tip.refine(self.npoints)

        # Scaling the profiles to the correct chord length.
        self.root.scale(self.chord[0])
        self.tip.scale(self.chord[1])

        # Rotating the profiles to allow for twist.
        rootCenter = deepcopy(self.root.get_qcpoint())
        tipCenter = deepcopy(self.tip.get_qcpoint())
        self.root.translate(rootCenter, -1)
        self.tip.translate(tipCenter, -1)
        self.root.rotate(yAxis, self.twist[0])
        self.tip.rotate(yAxis, self.twist[1])
        self.root.translate(rootCenter, 1)
        self.tip.translate(tipCenter, 1)

        # Adding the y-coordinate for span.
        self.root.translate(yAxis, self.span[0])
        self.tip.translate(yAxis, self.span[1])

        # Translating both airfoils for wing sweep.
        self.root.translate(xAxis, self.sweep[0])
        self.tip.translate(xAxis, self.sweep[1])

        # Normalise the position of the airfoil such the LE root is aligned with the origen
        self.root.translate(yAxis, -self.span[0])
        self.tip.translate(yAxis, -self.span[0])
        self.tip.translate(xAxis, -self.sweep[0])
        self.root.translate(xAxis, -self.sweep[0])

        # Rotating both foils for dihedral.
        self.root.rotate(xAxis, self.dihedral[0])
        self.tip.rotate(xAxis, self.dihedral[1])

    '''
    The align_le and align_qc functions rotate the section so the leading edge (le)
    or the rotation point (qc) are parallel to the y axis. This function allows wings
    with high sweep to be made on small machines and to minimize the amount of wasted
    foam.
    '''
    def align_le(self):
        # Find the angle of the leading edge
        xOffset = self.root.get_lepoint()[0] - self.tip.get_lepoint()[0]
        span = np.abs(self.span[0] - self.span[1])
        alpha = math.atan2(-xOffset, span)

        # Rotate the entire wing section build counterclockwise by angle alpha.
        self.root.rotate(zAxis, alpha)
        self.tip.rotate(zAxis, alpha)

    def align_qc(self):
        # Find the angle of the quarter chord line
        xOffset = self.root.get_qcpoint()[0] - self.tip.get_qcpoint()[0]
        span = np.abs(self.span[0] - self.span[1])
        alpha = math.atan2(-xOffset, span)

        # Rotate the entire wing section build counterclockwise by angle alpha.
        self.root.rotate(zAxis, alpha)
        self.tip.rotate(zAxis, alpha)

    '''
    Change the height of the current section to match the height of the previous section.
    This is helpfull in manufacturing.
    '''
    def height_alignment(self, Sec):
        # Find the difference in height of the tip qcpoint previous section and
        # the current root qcpoint height.
        diff = self.root.get_qcpoint()[1] - Sec.tip.get_qcpoint()[1]

        # Move current section up by the difference
        self.root.translate(zAxis, diff)
        self.tip.translate(zAxis, diff)

    '''
    This function translates the wing so that it is located in a user specified positions
    '''
    def locate_section(self):
        # Translating both airfoils for the root offset.
        translationVector = xAxis* self.xOffsetRoot + yAxis* self.yOffsetRoot + zAxis*self.zOffsetRoot
        self.root.translate(translationVector, 1)
        self.tip.translate(translationVector, 1)

'''
This class is responisble to convert the section profiles to cutting profiles and
to gcode. This class does depend on machine properties.
'''
class Profile:

    def __init__(self, Sec):

        self.Sec = Sec  # Must be an instance of Section.

        #Machine Properties.
        self.x_lenght = 600 # Maximum machine travel in the x-direction.
        self.z_lenght = 100 # Maximum machine travel in the z-direction.
        self.ySpan = 1000   # Distance between 2d movement platforms in the y direction.

        # Profiles to cut.
        self.rootCut = None # Wingroot profiles on parralel planes.
        self.tipCut = None  # Wingtip profiles on parralel planes.

        # Boundaries of the cut.
        self.xOffsetLE = 10 # Distance before cut profile.
        self.xOffsetTE = 10 # Distance after cut profile.
        self.yOffset = 10   # Distance above cut profile.

        # Main separated paths.
        self.rootTopPath = None
        self.rootBottomPath = None
        self.tipTopPath = None
        self.tipBottomPath = None
        self.rootOrigin = None
        self.tipOrigin = None

        '''
        # Cylindrical channel nner cutout.
        self.cylindricalChannel = True  # Set true if cutout must be included.
        self.rootChordPercentage = 0.3  # Distance from LE, in percentage chord.
        self.tipChordPercentage = 0.3   # Distance from LE, in percentage chord.
        self.rootDiameter = 10  # Diameter of the channel at the root.
        self.tipDiameter = 10   # Diameter of the channel at the tip.
        self.rootChannel = None # Coordinates to be cut root.
        self.tipChannel = None  # Coordinates to be cut tip.
        '''

        # Gcode file properties.
        self.fileName = 'test'  # Gcode file name.
        self.fileExtension = '.txt'  # Gcode filename extension.
        self.ax1 = 'X'  # 1st axis name.
        self.ax2 = 'Y'  # 2nd axis name.
        self.ax3 = 'U'  # 3th axis name.
        self.ax4 = 'Z'  # 4th axis name.
        self.gcode = [] # List containing the gcode commands.

        # Cutting properties.
        self.cuttingVoltage = 0   # Percentage of machine voltage send to the cutting wire.
        self.rapidFeed = 100   # Speed of the rapid parts of the operation, in mm/s.
        self.cuttingFeed = 100  # Speed of the cutting parts of the operation, in mm/s.
        self.rootKerf = 0   # Diameter of the cut.
        self.tipKerf = 0    # Diameter of the cut.

    def set_filename(fileName):
        self.fileName = fileName

    def set_yspan(self, ySpan):
        self.ySpan = ySpan

    def set_xoffset_le(self, xOffsetLE):
        self.xOffsetLE = xOffsetLE

    def set_xoffset_te(self, xOffsetTE):
        self.xOffsetTE = xOffsetTE

    def set_cutting_voltage(self, cuttingVoltage):
        self.cuttingVoltage = cuttingVoltage

    def set_rapid_feed(self, rapidFeed):
        self.rapidFeed = rapidFeed

    def set_cutting_feed(self, cuttingFeed):
        self.cuttingFeed = cuttingFeed

    def set_yoffset(self, yOffset):
        self.yOffset = yOffset

    def set_kerf(self, rootKerf, tipKerf):
        self.rootKerf = rootKerf
        self.tipKerf = tipKerf

    def get_yspan(self):
        return self.ySpan

    def get_profiles(self):
        return self.rootCut, self.tipCut

    '''
    This function find the projected shapes intersecting with a plane.
    This is achieved by drawing a straign line throuhg the n-th point on both the
    tip and root foil and finding the intersection of that line with the cutting planes.
    '''
    def project(self, plane):
        # Get airfoil coordinates
        rootFoil, tipFoil = self.Sec.get_foils()

        # Find directional ratios
        directionalRat = tipFoil - rootFoil

        # Intersection with cutting planes y
        constants = (plane - rootFoil[:, 1])/directionalRat[:, 1]

        # X coordinates
        xProjection = constants * directionalRat[:, 0] + rootFoil[:, 0]

        # Z coordinates
        zProjection = constants * directionalRat[:, 2] + rootFoil[:, 2]

        # Y coordinates
        y = np.zeros(len(zProjection))+plane

        # Store the coordinates in two numpy arrays
        return np.array([xProjection, y, zProjection]).T

    '''
    Project the wing geometry on the cutting planes.
    '''
    def cutting_planes(self):
        self.rootCut = self.project(0)[:-1,:]
        self.tipCut = self.project(self.ySpan)[:-1,:]

    '''
    Split the geometry into distinct parts that can be cut.
    '''
    def paths(self):
        # Get a 2d array for the cutting profile.
        xRoot = self.rootCut[:,0]
        yRoot = self.rootCut[:,2]
        root = np.array([xRoot,yRoot]).T

        xTip = self.tipCut[:,0]
        yTip = self.tipCut[:,2]
        tip = np.array([xTip,yTip]).T

        # Split the profile in a top and bottom side.
        rootTop, rootBottom = np.split(root, 2, 0)
        tipTop, tipBottom = np.split(tip, 2, 0)

        # Compensate for kerf width
        rootTop = self.kerf_compensation_airfoil(rootTop, self.rootKerf)
        rootBottom = self.kerf_compensation_airfoil(rootBottom, self.rootKerf)

        tipTop = self.kerf_compensation_airfoil(tipTop, self.tipKerf)
        tipBottom = self.kerf_compensation_airfoil(tipBottom, self.tipKerf)

        # Find highest point of profiles
        rootMax = np.amax(rootTop[:,1])
        tipMax = np.amax(tipTop[:,1])
        yOffset = max(rootMax, tipMax) + self.yOffset

        # Set the respective origin a certain offset from leading edge up and to the left
        self.rootOrigin = np.array([rootTop[-1][0] - self.xOffsetLE, yOffset])
        self.tipOrigin = np.array([tipTop[-1][0] - self.xOffsetLE, yOffset])

        # alignment mode, the wire will be at the at the exact point where the foam needs to be positioned.
        self.rootAlign = np.array([rootTop[-1][0] - self.xOffsetLE, rootMax])
        self.tipAlign = np.array([tipTop[-1][0] - self.xOffsetLE, tipMax])

        # Make the lead in paths.
        tr = np.array([rootTop[-1][0] - self.xOffsetLE, rootTop[-1][1]])
        tt = np.array([tipTop[-1][0] - self.xOffsetLE, tipTop[-1][1]])
        br = np.array([rootBottom[0][0] - self.xOffsetLE, rootBottom[0][1]])
        bt = np.array([tipBottom[0][0] - self.xOffsetLE, tipBottom[0][1]])

        rootBottom = np.append([br], rootBottom, 0)
        rootTop = np.append(rootTop, [tr], 0)

        tipBottom = np.append([bt], tipBottom, 0)
        tipTop = np.append(tipTop, [tt], 0)

        # Make the lead out paths.
        tr = np.array([rootTop[0][0] + self.xOffsetTE, rootTop[0][1]])
        tt = np.array([tipTop[0][0] + self.xOffsetTE, tipTop[0][1]])
        br = np.array([rootBottom[-1][0] + self.xOffsetTE, rootBottom[-1][1]])
        bt = np.array([tipBottom[-1][0] + self.xOffsetTE, tipBottom[-1][1]])

        rootTop = np.append([tr], rootTop, 0)
        tipTop = np.append([tt], tipTop, 0)

        rootBottom = np.append(rootBottom, [br], 0)
        tipBottom = np.append(tipBottom, [bt], 0)

        # Flip bottom array such that it starts from trailing edge to leading edge.
        rootBottom = np.flip(rootBottom, 0)
        tipBottom = np.flip(tipBottom, 0)

        self.rootTopPath = rootTop
        self.rootBottomPath = rootBottom
        self.tipTopPath = tipTop
        self.tipBottomPath = tipBottom

        # Plot the paths
        fig = plt.figure('Airfoils', figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(rootTop[:,0], rootTop[:,1], marker = '.', color = 'r')
        ax.plot(rootBottom[:,0], rootBottom[:,1], marker = '.', color = 'r')
        ax.plot(tipTop[:,0], tipTop[:,1], marker = '.', color = 'b')
        ax.plot(tipBottom[:,0], tipBottom[:,1], marker = '.', color = 'b')

        ax.plot(xRoot, yRoot, color = 'k')
        ax.plot(xTip, yTip, color = 'k')
        plt.show()

    '''
    Add an offset to the paths with distance half the kerf diameter.
    This compensates for cutting diameter.
    Only valid for small diameter wires.
    '''
    def kerf_compensation_airfoil(self, path, kerf):
        # Difference line segment vectors.
        diff = (np.pad(path, ((0, 1), (0, 0)), 'constant') - np.pad(path, ((1, 0), (0, 0)), 'constant'))[1:-1,:]

        # Find angle of each line segment.
        angle = np.arctan2(diff[:,1], diff[:,0])

        # Convert angle to unit vector, rotated 90 deg ccw from original.
        x = np.sin(angle)
        y = -np.cos(angle)
        unit = np.array([x, y]).T

        # Take the sum of the neighbouring unit vectors.
        direction = np.pad(unit, ((0, 1),(0, 0)), 'edge') + np.pad(unit, ((0, 1),(0, 0)), 'edge')

        # Normalize vector length to have half the kerf diameter as length.
        magnitude = np.linalg.norm(direction, axis=1)
        magnitude = np.array([magnitude, magnitude]).T
        translate = direction/magnitude * kerf/2

        # Translate each point to the new location
        newPath = translate + path
        return newPath

    """
    Converts list of coords to gcode
    G0 = fast movement, goes to location as fast as possible
    G1 = cutting at specified feedrate (In combination with F)
    F = feed rate in mm/min
    M5 = turn off hot wire
    M3 = turn on hot wire (in combination with S)
    S = percentage of voltage to hot wire
    M0 = pause program until cycle start command (~)
    ~ = cycle start
    """
    def coords_to_gcode(self, directory, mirror = False):
        # Check if file name already exists, increment file name if needed.
        fileName = f'{directory}/{self.fileName}{self.fileExtension}'
        name = self.fileName
        i = 1
        while os.path.exists(fileName):
            fileName = f'{directory}/{self.fileName}({i}){self.fileExtension}'
            name = f'{self.fileName}({i})'
            i += 1

        if mirror:
            fileName = f'{directory}/{self.fileName}_mirror{self.fileExtension}'
            name = f'{self.fileName}_mirror'
            i = 1
            while os.path.exists(fileName):
                fileName = f'{directory}/{self.fileName}_mirror({i}){self.fileExtension}'
                name = f'{self.fileName}_mirror({i})'
                i += 1
        self.gcode = ['%','('+self.fileName+') G21 G90']

        # Converts the tip and root numpy arrays to machine readable gcode.
        def numpy_to_line(root, tip):
            if mirror:
                return f'{self.ax1}{root[0]} {self.ax2}{root[1]} {self.ax3}{tip[0]} {self.ax4}{tip[1]}'
            else:
                return f'{self.ax1}{tip[0]} {self.ax2}{tip[1]} {self.ax3}{root[0]} {self.ax4}{root[1]}'

        # Sets the speed for rapid movement.
        self.gcode.append(f'G1 F{self.rapidFeed}')

        # Go to alignment position.
        # Pause the machine to allow for precise positioning of the foam.
        # The program will continue with a cycle start command.
        self.gcode.append(numpy_to_line(self.rootAlign, self.tipAlign))
        self.gcode.append('M5 M0')

        # Move straight up to origin.
        # Pause the machine to allow for checking the height.
        # The program will continue with a cycle start command.
        self.gcode.append(numpy_to_line(self.rootOrigin, self.tipOrigin))
        self.gcode.append('M5 M0')

        # Move horizontally untill x coordinate TE cut.
        rootPoint = np.array([self.rootTopPath[0, 0], self.rootOrigin[1]])
        tipPoint = np.array([self.tipTopPath[0, 0], self.tipOrigin[1]])
        self.gcode.append(numpy_to_line(rootPoint, tipPoint))

        # Top airfoil cut
        self.gcode.append(f'G1 F{self.cuttingFeed} M3 S{self.cuttingVoltage}')
        for i in range(len(self.rootTopPath)):
            rootPoint = np.array([self.rootTopPath[i, 0], self.rootTopPath[i, 1]])
            tipPoint = np.array([self.tipTopPath[i, 0], self.tipTopPath[i, 1]])
            self.gcode.append(numpy_to_line(rootPoint, tipPoint))

        # Back to origin points and pause machine for removal top foam.
        self.gcode.append(numpy_to_line(self.rootOrigin, self.tipOrigin))
        self.gcode.append('M5 M0')

        # Set rapid speed
        self.gcode.append(f'G1 F{self.rapidFeed}')

        # Move horizontally untill x coordinate TE cut.
        rootPoint = np.array([self.rootTopPath[0, 0], self.rootOrigin[1]])
        tipPoint = np.array([self.tipTopPath[0, 0], self.tipOrigin[1]])
        self.gcode.append(numpy_to_line(rootPoint, tipPoint))

        # Bottom airfoil cut.
        self.gcode.append(f'G1 F{self.cuttingFeed} M3 S{self.cuttingVoltage}')
        for i in range(len(self.rootBottomPath)):
            rootPoint = np.array([self.rootBottomPath[i, 0], self.rootBottomPath[i, 1]])
            tipPoint = np.array([self.tipBottomPath[i, 0], self.tipBottomPath[i, 1]])
            self.gcode.append(numpy_to_line(rootPoint, tipPoint))

        # Back to origin points
        self.gcode.append(numpy_to_line(self.rootOrigin, self.tipOrigin))

        # Set rapid speed and turn of wire.
        self.gcode.append(f'G1 F{self.rapidFeed} M5')

        # Go back to zero.
        self.gcode.append(f'{self.ax1}0 {self.ax2}0 {self.ax3}0 {self.ax4}0')

        # End gcode.
        self.gcode.append(f'%')

        # Write gcode to file.
        gcode = '\n'.join(self.gcode)
        with open(fileName, 'w') as f:
            f.write(gcode)
