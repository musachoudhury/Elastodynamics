import gmsh
import sys

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("L Block Mesh")

# Define L-block dimensions and mesh density
Lx, Ly, Lz = 2.0, 2.0, 1.0  # Outer dimensions of the block
Tx, Ty = 1.0, 1.0           # Thickness of the L shape (cut-out dimensions)
nx, ny, nz = 20, 20, 10     # Number of divisions in x, y, z directions

# Define element sizes
lc_x = Lx / nx
lc_y = Ly / ny
lc_z = Lz / nz

# Outer box points
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc_x)
p2 = gmsh.model.geo.addPoint(Lx, 0, 0, lc_x)
p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, lc_x)
p4 = gmsh.model.geo.addPoint(0, Ly, 0, lc_x)
p5 = gmsh.model.geo.addPoint(0, 0, Lz, lc_x)
p6 = gmsh.model.geo.addPoint(Lx, 0, Lz, lc_x)
p7 = gmsh.model.geo.addPoint(Lx, Ly, Lz, lc_x)
p8 = gmsh.model.geo.addPoint(0, Ly, Lz, lc_x)

# Inner box (cut-out) points
p9 = gmsh.model.geo.addPoint(Tx, Ty, 0, lc_x)
p10 = gmsh.model.geo.addPoint(Lx, Ty, 0, lc_x)
p11 = gmsh.model.geo.addPoint(Tx, Ly, 0, lc_x)
p12 = gmsh.model.geo.addPoint(Tx, Ty, Lz, lc_x)
p13 = gmsh.model.geo.addPoint(Lx, Ty, Lz, lc_x)
p14 = gmsh.model.geo.addPoint(Tx, Ly, Lz, lc_x)

# Define outer box surfaces
s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([
    gmsh.model.geo.addLine(p1, p2),
    gmsh.model.geo.addLine(p2, p3),
    gmsh.model.geo.addLine(p3, p4),
    gmsh.model.geo.addLine(p4, p1)
])])

# Create surfaces for the rest of the outer and inner shapes
# Add other surfaces for the complete volume...

# Sync geometry
gmsh.model.geo.synchronize()

# Mesh generation parameters
gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear elements
gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Default 3D meshing algorithm
gmsh.option.setNumber("Mesh.Optimize", 1)     # Optimize the mesh

# Generate the 3D mesh
gmsh.model.mesh.generate(3)

# Save the mesh to a file
gmsh.write("L_block_mesh.msh")

# Launch the GUI (optional)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize Gmsh
gmsh.finalize()
