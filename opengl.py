import numpy as np
from glumpy import app, gloo, gl, glm

# Load the coordinate dump and transpose it so we have a list
# of coordinates, each coordinate having 3 points
# TODO: Probably move transpose into the thing that saves it
coordinate_dump = np.load("coordinate_dump.npz")
root = coordinate_dump['root'].T
tip = coordinate_dump['tip'].T

print(root)

window = app.Window()

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
} """

fragment = """
void main()
{
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
} """

#root = np.array([[0, 0, 0], [100, 100, 100]])

V = np.zeros(root.shape[0] + tip.shape[0], [("a_position", np.float32, 3)])
V["a_position"] = np.vstack((root, tip)) / 100

I_list = []
for i in range(root.shape[0]):
    # Connect the airfoil segments themselves
    I_list.append([i, (i + 1) % (root.shape[0] - 1)])

    # Connect two airfoils
    I_list.append([i, i + root.shape[0]])
I = np.array(I_list, dtype=np.uint32)
print(V)
print(I)

#V["a_position"] = [[ 1, 1, 1], [-1, 1, 1], [-1,-1, 1], [ 1,-1, 1],
#                   [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], [-1,-1,-1]]
#I = np.array([0,1,2, 0,2,3,  0,3,4, 0,4,5,  0,5,6, 0,6,1,
#              1,6,7, 1,7,2,  7,4,3, 7,3,2,  4,7,6, 4,6,5], dtype=np.uint32)

#I = np.array([0,1,2, 0,2,3,  0,3,4, 0,4,5,  0,5,6, 0,6,1,
#              1,6,7, 1,7,2,  7,4,3, 7,3,2,  4,7,6, 4,6,5], dtype=np.uint32)

V = V.view(gloo.VertexBuffer)
I = I.view(gloo.IndexBuffer)

cube = gloo.Program(vertex, fragment)
cube["a_position"] = V

view = np.eye(4,dtype=np.float32)
model = np.eye(4,dtype=np.float32)
projection = np.eye(4,dtype=np.float32)
glm.translate(view, 0,0,-5)
cube['u_model'] = model
cube['u_view'] = view
cube['u_projection'] = projection
phi, theta = 0,0

@window.event
def on_resize(width, height):
   ratio = width / float(height)
   cube['u_projection'] = glm.perspective(45.0, ratio, 2.0, 100.0)

@window.event
def on_draw(dt):
    global phi, theta
    window.clear()
    cube.draw(gl.GL_LINES, I)

    # Make cube rotate
    theta += 0.5 # degrees
    phi += 0.5 # degrees
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, theta, 0, 0, 1)
    glm.rotate(model, phi, 0, 1, 0)
    cube['u_model'] = model

@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)

# Run the app
app.run()