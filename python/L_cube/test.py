# %%
#This code is based off of Jeremy Bleyers Transient elastodynamics with Newmark time-integration tutorial
#https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output

from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh
import gmsh
from dolfinx.io.gmshio import read_from_msh


gdim = 3  # domain geometry dimension
fdim = gdim-1  # facets dimension

gmsh.initialize()

occ = gmsh.model.occ
mesh_comm = MPI.COMM_WORLD
model_rank = 0

domain, cell_tags, facet_tags = read_from_msh('Lshapemesh.msh', mesh_comm, gdim=gdim)
      
# %%
