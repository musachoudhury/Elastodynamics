import gmsh
import sys
import math

gmsh.initialize(sys.argv)

gmsh.model.add("boxmesh")

lc = 0.25

h = 50/1000
w = 100/1000/2
l = 100/1000/2
a = 25/1000
b = 25/1000
q = 3

factory = gmsh.model.geo

factory.addPoint(0, 0, 0, lc, 1)
factory.addPoint(w, 0, 0, lc, 2)
factory.addPoint(w, l, 0, lc, 3)
factory.addPoint(0, l, 0, lc, 4)
factory.addPoint(0, 0, h, lc, 5)
factory.addPoint(w, 0, h, lc, 6)
factory.addPoint(w, l-b, h, lc, 7)
factory.addPoint(w, l, h, lc, 8)
factory.addPoint(w-a, l, h, lc, 9)
factory.addPoint(0, l, h, lc, 10)
factory.addPoint(w-a, l-b, h, lc, 11)


#bottom face
factory.addLine(1, 2, 1)
factory.addLine(2, 3, 2)
factory.addLine(3, 4, 3)
factory.addLine(4, 1, 4)

#top face
factory.addLine(5, 6, 5)
factory.addLine(6, 7, 6)
factory.addLine(7, 8, 7)
factory.addLine(8, 9, 8)
factory.addLine(9, 10, 9)
factory.addLine(10, 5, 10)

factory.addLine(9, 11, 11)
factory.addLine(11, 7, 12)

#vertical
factory.addLine(1, 5, 13)
factory.addLine(2, 6, 14)
factory.addLine(3, 8, 15)
factory.addLine(4, 10, 16)

curve_loops = [
    factory.addCurveLoop([1, 2, 3, 4], 1),             #bottom 
    factory.addCurveLoop([5, 6, -12, -11, 9, 10], 2), #top L-shape
    factory.addCurveLoop([1, 14, -5, -13], 3),          #front
    factory.addCurveLoop([-3, 15, 8, 9, -16], 4),         #back
    factory.addCurveLoop([-4, 16, 10, -13], 5),          #left
    factory.addCurveLoop([2, 15, -7, -6, -14], 6),         #right
    factory.addCurveLoop([12, 7, 8, 11], 7)          #load surface
]

surfaces = [factory.addPlaneSurface([loop]) for loop in curve_loops]

# Create surface loop and volume
surface_loop = factory.addSurfaceLoop(surfaces)
volume = factory.addVolume([surface_loop])

factory.synchronize()

gmsh.model.addPhysicalGroup(3, [volume], 1)  # Volume

# Add boundary markers
# gmsh.model.addPhysicalGroup(2, [surfaces[0]], name="bottom")
# gmsh.model.addPhysicalGroup(2, [surfaces[1]], name="top")
# gmsh.model.addPhysicalGroup(2, [surfaces[2]], name="front")
# gmsh.model.addPhysicalGroup(2, [surfaces[3]], name="back")
# gmsh.model.addPhysicalGroup(2, [surfaces[4]], name="left")
# gmsh.model.addPhysicalGroup(2, [surfaces[5]], name="right")
# gmsh.model.addPhysicalGroup(2, [surfaces[6]], name="load_surface")

#gmsh.model.addPhysicalGroup(3, [volume], name="body")

gmsh.model.mesh.generate(3)
gmsh.write("boxmesh.msh")


if '-nopopup' not in sys.argv:
    gmsh.fltk.run()


