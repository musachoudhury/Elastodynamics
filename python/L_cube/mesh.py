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
factory.addPoint(w, l, 0, lc, 3)
factory.addPoint(0, l, 0, lc, 4)

factory.addPoint(w2, 0, h2, lc, 5)
factory.addPoint(w, 0, h2, lc, 6)
factory.addPoint(w, l, h2, lc, 7)
factory.addPoint(w2, l, h2, lc, 8)

factory.addPoint(0, 0, h, lc, 9)
factory.addPoint(w2, 0, h, lc, 10)
factory.addPoint(w2, l, h, lc, 11)
factory.addPoint(0, l, h, lc, 12)

#bottom face
l1 = factory.addLine(1, 2, 1)
l2 = factory.addLine(2, 3, 2)
l3 = factory.addLine(3, 4, 3)
l4 = factory.addLine(4, 1, 4)

#top face 1
l5 = factory.addLine(9, 10, 5)
l6 = factory.addLine(10, 11, 6)
l7 = factory.addLine(11, 12, 7)
l8 = factory.addLine(12, 9, 8)

#top face 2
l9 = factory.addLine(5, 6, 9)
l10 = factory.addLine(6, 7, 10)
l11 = factory.addLine(7, 8, 11)
l12 = factory.addLine(8, 5, 12)

#vertical 1
l13 = factory.addLine(1, 9, 13)
l14 = factory.addLine(4, 12, 14)

#vertical 2
l15 = factory.addLine(5, 10, 15)
l16 = factory.addLine(8, 11, 16)

#vertical 3
l17 = factory.addLine(2, 6, 17)
l18 = factory.addLine(3, 7, 18)

curve_loops = [
    factory.addCurveLoop([1, 2, 3, 4], 1),             #bottom 
    factory.addCurveLoop([5, 6, 7, 8], 2),             #top 1
    factory.addCurveLoop([9, 10, 11, 12], 3),         #top 2
    factory.addCurveLoop([1, 17, -9, 15, -5, -13], 4),         #front
    factory.addCurveLoop([-3, 18, 11, 16, 7, -14], 5),         #back
    factory.addCurveLoop([-4, 14, 8, -13], 6),          #left
    factory.addCurveLoop([2, 18, -10, -17], 7),         #right 1
    factory.addCurveLoop([-12, 16, -6, -15], 8),         #right 2
]

surfaces = [factory.addPlaneSurface([loop]) for loop in curve_loops]

# gmsh.model.geo.mesh.setTransfiniteSurface(surfaces[6], "right1", [5, 8, 10, 11])
surface = factory.addPlaneSurface([curve_loops[3]])

# Create surface loop and volume
surface_loop = factory.addSurfaceLoop(surfaces)


volume = factory.addVolume([surface_loop])

factory.synchronize()

gmsh.model.addPhysicalGroup(3, [volume], 1)  # Volume

# Add boundary markers
gmsh.model.addPhysicalGroup(2, [surfaces[0]], name="bottom")
gmsh.model.addPhysicalGroup(2, [surfaces[1]], name="top 1")
gmsh.model.addPhysicalGroup(2, [surfaces[2]], name="top 2")
gmsh.model.addPhysicalGroup(2, [surfaces[3]], name="front")
gmsh.model.addPhysicalGroup(2, [surfaces[4]], name="back")
gmsh.model.addPhysicalGroup(2, [surfaces[5]], name="left")
gmsh.model.addPhysicalGroup(2, [surfaces[6]], name="right 1")
gmsh.model.addPhysicalGroup(2, [surfaces[7]], name="right 2")


gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.RecombineAll", 0)

gmsh.model.mesh.generate(3)
gmsh.write("Lshapemesh.msh")





if '-nopopup' not in sys.argv:
    gmsh.fltk.run()



# %%
