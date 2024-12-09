import gmsh
import sys 

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("Box")

# Define the dimensions of the box
length = 1.0  # Length in x-direction
width = 1.0   # Width in y-direction
height = 1.0  # Height in z-direction

# Create a box
gmsh.model.occ.addBox(0, 0, 0, length, width, height)

# Synchronize to reflect the changes
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.Algorithm", 11)
gmsh.model.mesh.recombine()
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)


# Generate a 3D mesh
gmsh.model.mesh.generate(3)

# gmsh.model.mesh.refine()
# Save the mesh to a file
#gmsh.write("box.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize Gmsh
gmsh.finalize()
