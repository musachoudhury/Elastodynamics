#This code is based off of Jeremy Bleyers Hyperelasticity tutorial
#https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html

import numpy as np
import ufl
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import fem, io, nls, mesh, default_scalar_type, log
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from dolfinx.mesh import create_box, CellType
from ufl import (
    as_matrix,
    dot,
    cos,
    sin,
    SpatialCoordinate,
    Identity,
    grad,
    ln,
    tr,
    det,
    variable,
    derivative,
    TestFunction,
    TrialFunction,
    inner,
    FacetNormal
)
from dolfinx.io.gmshio import read_from_msh
import gmsh


gdim = 3  # domain geometry dimension
fdim = 2  # facets dimension

mesh_comm = MPI.COMM_WORLD
model_rank = 0

h = 50.0
w = 100/2
l = 100/2

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([w, l, h])],
                         [4, 4, 4], cell_type=mesh.CellType.hexahedron)



dim = domain.topology.dim

print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)
V = fem.functionspace(domain, ("P", degree, shape))

u = fem.Function(V, name="Displacement")

# Identity tensor
Id = Identity(dim)

# Deformation gradient
F = variable(Id + grad(u))

# Right Cauchy-Green tensor
C = F.T * F

# Invariants of deformation tensors
I1 = tr(C)
J = det(F)
E_GL = variable(0.5*(C-Id))
# Shear modulus
E = 1e4
nu = 0.4
#mu = fem.Constant(domain, E / 2 / (1 + nu))
#lmbda = fem.Constant(domain, E * nu / (1 - 2 * nu) / (1 + nu))

lmbda = fem.Constant(domain, 499.92568)
mu = fem.Constant(domain, 1.61148)

q = 3.0

# Stored strain energy density (compressible neo-Hookean model)
psi = mu / 2 * (I1 - 3 - 2 * ln(J)) + lmbda / 2 * (J - 1) ** 2

# Stored strain energy density (St. Venant-Kirchhoff model)
#psi = lmbda/2 * tr(E_GL)**2 + mu * tr(E_GL*E_GL)

# PK1 stress = d_psi/d_F
P = ufl.diff(psi, F)


def bottom(x):
    return np.isclose(x[2], 0)

def top(x):
    return np.isclose(x[2], h)

def right(x):
    return np.isclose(x[0], w)

def back(x):
    return np.isclose(x[1], l)

def load_surface(x):
    return np.logical_and(np.logical_and(x[0] >= w/2, x[1] >= l/2), np.isclose(x[2],  h))

load_facets = mesh.locate_entities_boundary(
    domain, fdim, load_surface)

#add tags to facets for the load surface
marked_facets = np.hstack([load_facets])
marked_values = np.hstack([np.full_like(load_facets, 1)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

#locate facets 
bottom_facets = mesh.locate_entities_boundary(
    domain, fdim, bottom)

top_facets = mesh.locate_entities_boundary(
    domain, fdim, top)

right_facets = mesh.locate_entities_boundary(
    domain, fdim, right)

back_facets = mesh.locate_entities_boundary(
    domain, fdim, back)

#Identify dof subcomponents  
bottom_dofs_z = fem.locate_dofs_topological(V.sub(2), fdim, bottom_facets)
top_dofs_u = fem.locate_dofs_topological(V.sub(0), fdim, top_facets)
top_dofs_v = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)
right_dofs_u = fem.locate_dofs_topological(V.sub(0), fdim, top_facets)
back_dofs_v = fem.locate_dofs_topological(V.sub(1), fdim, back_facets)

#Boundary conditions
bcs = [
    fem.dirichletbc(default_scalar_type(0), bottom_dofs_z, V.sub(2)),
    fem.dirichletbc(default_scalar_type(0), top_dofs_u, V.sub(0)),
    fem.dirichletbc(default_scalar_type(0), top_dofs_v, V.sub(1)),
    #Symmetry conditions
    fem.dirichletbc(default_scalar_type(0), right_dofs_u, V.sub(0)),
    fem.dirichletbc(default_scalar_type(0), back_dofs_v, V.sub(1))
]

dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

v = TestFunction(V)
du = TrialFunction(V)

#E_pot = psi * dx

# Residual = derivative(
#     E_pot, u, v
# )  # This is equivalent to Residual = inner(P, grad(v))*dx

T = fem.Constant(domain, q)
n = FacetNormal(domain)
Load = dot(T * n, v) * ds(1)

Residual = inner(P, grad(v))*dx + Load
Jacobian = derivative(Residual, u, du)

problem = fem.petsc.NonlinearProblem(Residual, u, bcs)

solver = nls.petsc.NewtonSolver(domain.comm, problem)
# Set Newton solver options
solver.atol = 1e-4
solver.rtol = 1e-4
solver.ksp_type = "preonly"
solver.pc_type = "lu"

solver.convergence_criterion = "incremental"

out_file = "hyperelasticity.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)

u.vector.set(0.0)

def sigma(v):
    return 1/J*F*P


def stiffness(u, u_):
    return inner(1/J*F*P, grad(u_))*dx

E_el = fem.form(0.5 * stiffness(u, u))

log.set_log_level(log.LogLevel.INFO)

num_its, converged = solver.solve(u)
assert converged

u.x.scatter_forward()  # updates ghost values for parallel computations

s = sigma(u) - 1. / 3 * ufl.tr(sigma(u)) * ufl.Identity(len(u))
von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises, name="Stress")
stresses.interpolate(stress_expr)

vtk = io.VTKFile(domain.comm, "hyperelasticity.pvd", "w")
vtk.write_function(u, 0)
vtk.write_function(stresses, 0)
vtk.close()