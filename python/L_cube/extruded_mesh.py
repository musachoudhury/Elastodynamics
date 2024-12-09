#%%
import gmsh
import sys

gmsh.initialize(sys.argv)

gmsh.model.add("Lshapemesh")
#gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)
lc = 1.0

w = 6
l = 3
h = 10

w2 = 3
h2 = 3

# a = 25/1000
# b = 25/1000
# q = 3

factory = gmsh.model.geo

factory.addPoint(0, 0, 0, lc, 1)
factory.addPoint(w, 0, 0, lc, 2)
factory.addPoint(w, 0, h2, lc, 3)
factory.addPoint(w-w2, 0, h2, lc, 4)

factory.addPoint(w-w2, 0, h, lc, 5)
factory.addPoint(0, 0, h, lc, 6)

#bottom face
l1 = factory.addLine(1, 2, 1)
l2 = factory.addLine(2, 3, 2)
l3 = factory.addLine(3, 4, 3)
l4 = factory.addLine(4, 5, 4)

l5 = factory.addLine(5, 6, 5)
l6 = factory.addLine(6, 1, 6)

gmsh.model.geo.mesh.setTransfiniteCurve(l1, 7)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, 4)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, 4)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, 8)
gmsh.model.geo.mesh.setTransfiniteCurve(l5, 4)
gmsh.model.geo.mesh.setTransfiniteCurve(l6, 11)

curve_loop = factory.addCurveLoop([1, 2, 3, 4, 5, 6], 1)

surface = factory.addPlaneSurface([curve_loop])


ov = gmsh.model.geo.extrude([(2, surface)], 0, 3, 0)

factory.synchronize()

gmsh.model.addPhysicalGroup(3, [ov[0][1]], 1)  # Volume

gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.RecombineAll", 0)

gmsh.model.mesh.generate(3)
gmsh.write("Lshapemesh.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()