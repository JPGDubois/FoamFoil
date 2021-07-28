# FoamFoil
4-axis G-code generator for cutting out wing sections with homebuilt CNC hotwirecutters.


Currently most files are under development. 

foil_refine.py (debugged and functional) deals with airfoil refinement and 3d geometry setup. 
To use it you instantiate a section by MySection = foil_refine.Section(self, <root_foil_array>, <tip_foil_array>)

Then the instance's setters can be used to set all the geometric variables (span, root chord, tip chord, sweep, dihedral, twist, etc.).
  
Then you simply call MySection.build(self) and instantiate MyCoordinates = foil_refine.Coordinates(self, MyObjectName)

To get the gcode (not yet tested, likely needs debugging) you then need coords_to_gcode.py
You need to instantiate MyGcode = coords_to_gcode.Gcode(self, MyCoordinates)
You can use setters to set the cut direction (currently only 'te' implemented)

To generate the code, simply call MyGcode.build(self). Then MyGcode.get_gcode(self) returns the gcode as an array of strings.

This all is meant to happen inside a GUI file that is not implemented yet.

(All other files are either works in progress, will be deprecated, or were developed by contributors other than Eduardo and perform other functions)

