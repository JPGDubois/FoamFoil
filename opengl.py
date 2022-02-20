import numpy as np
import math
from glumpy import app, gloo, gl, glm

# Load the coordinate dump and transpose it so we have a list
# of coordinates, each coordinate having 3 points
# TODO: Probably move transpose into the thing that saves it
coordinate_dump = np.load("coordinate_dump.npz")
root = coordinate_dump['root'].T
tip = coordinate_dump['tip'].T


window = app.Window()

# Set up shader code that enables us to use projection on vertices
vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
} """

# Shader code that specified the color for the lines that make up the wing wireframe
fragment = """
void main()
{
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
} """

# Shader code that specifies the color for the reference plane
plane_fragment = """
void main()
{
    gl_FragColor = vec4(1.0, 0.0, 1.0, 0.3);
} """

# Load in all airfoil points/vertices
V = np.zeros(root.shape[0] + tip.shape[0], [("a_position", np.float32, 3)])
# Scale by a factor of 100, which seems to somewhat normalize it to what fits on a screen. Also offset it to try
# to get the center of rotation to be somewhat in the middle (I can probably do an exact solution, but laziness you
# know)
V["a_position"] = (np.vstack((root, tip)) - np.array([50, 200, 50])) / 100

# Connect the airfoil segments using straight lines
# NOTE: Assumes number of tip/root data points are the same
I_list = []
for i in range(0, root.shape[0]):
    # Connect the airfoil segments themselves
    I_list.append([i, (i + 1) % (root.shape[0] - 1)])
    I_list.append([i + root.shape[0], (i + root.shape[0] + 1) % (root.shape[0] + tip.shape[0] - 1)])

    # Connect two airfoils (but only do this for some data points to avoid weird rendering effects due to a large
    # number of lines)
    if i % 5 == 0:
        I_list.append([i, i + root.shape[0]])
# Convert the line connections into a numpy array; we could technically calculate the size, but sinced this is a one-off
# process I'm not too worries about efficiency
I = np.array(I_list, dtype=np.uint32)

# Send the wing vertex/index buffer information off to the rendering engine so we can draw it to the screen later
V = V.view(gloo.VertexBuffer)
I = I.view(gloo.IndexBuffer)
cube = gloo.Program(vertex, fragment)
cube["a_position"] = V

# Now create a simple plane located roughly at the root of the wing so we have a sense of direction when panning around
V_plane = np.zeros(4, [("a_position", np.float32, 3)])
V_plane["a_position"] = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]]) * 2 + np.array([0, 0, -2]) - root[0] / 100
I_plane = np.array([0, 1, 2,  0, 3, 2], dtype=np.uint32)
V_plane = V_plane.view(gloo.VertexBuffer)
I_plane = I_plane.view(gloo.IndexBuffer)
plane = gloo.Program(vertex, plane_fragment)
plane["a_position"] = V_plane

view = np.eye(4,dtype=np.float32)
model = np.eye(4,dtype=np.float32)
projection = np.eye(4,dtype=np.float32)
glm.translate(view, 0,0,-5)
cube['u_model'] = model
cube['u_view'] = view
cube['u_projection'] = projection

plane['u_model'] = model
plane['u_view'] = view
plane['u_projection'] = projection

phi, theta = 70, 55


@window.event
def on_resize(width, height):
    ratio = width / float(height)
    cube['u_projection'] = glm.perspective(45.0, ratio, 2.0, 100.0)
    plane['u_projection'] = glm.perspective(45.0, ratio, 2.0, 100.0)


draw_time = 5  # s
cur_time = 0  # s; total elapsed time
@window.event
def on_draw(dt):
    global phi, theta, cur_time

    # Clear the window and draw the wing, reference plane.
    # NOTE: We're gradually showing nore of the airfoil, which is why we only select some of the points. This does
    # assume that there is some sense of order to the points / they form a kind of sequence
    window.clear()
    cube.draw(gl.GL_LINES, I[:math.ceil((cur_time % draw_time) / draw_time * len(I))])
    plane.draw(gl.GL_TRIANGLES, I_plane)

    # Increase the time counter and update the projection matrices to match the requested pan angles
    cur_time += dt
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, theta, 0, 0, 1)
    glm.rotate(model, phi, 0, 1, 0)
    cube['u_model'] = model
    plane['u_model'] = model


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)


# Keyboard input handlers. They do not appear to create smooth movement, but allow basic panning using the arrow keys
# NOTE: Unable to find a whole lot of documentation; guessing the variables at this point
@window.event
def on_key_press(keycode, something_else):
    global phi, theta
    print(keycode)
    if keycode == 65363:
        phi += 1
    elif keycode == 65361:
        phi -= 1
    elif keycode == 65362:
        theta += 1
    elif keycode == 65364:
        theta -= 1


# Run the app
app.run()
