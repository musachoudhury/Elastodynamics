import gmsh
import sys

gmsh.initialize()

gmsh.model.add("t16")

gmsh.logger.start()
gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)

w = 6
l = 3
h = 10

w2 = 3
h2 = 3

gmsh.model.occ.addBox(0, 0, 0, w, l, h, 1)
gmsh.model.occ.addBox(w2, 0, h2, w, l, h, 2)

ov, ovv = gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3)


gmsh.model.occ.synchronize()


gmsh.model.addPhysicalGroup(3, [3], 1)

#gmsh.option.setNumber("Mesh.Algorithm3D", 2)  # Delaunay algorithm

gmsh.model.mesh.generate(3)
gmsh.model.mesh.recombine()
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
gmsh.model.mesh.refine()
gmsh.write("t16.msh")

log = gmsh.logger.get()
print("Logger has recorded " + str(len(log)) + " lines")
gmsh.logger.stop()

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
