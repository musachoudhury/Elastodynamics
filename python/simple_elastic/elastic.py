#%%
# Scaled variable
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
import utils

L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

def clamped_boundary(x):
    return np.isclose(x[0], 0)

def load_surface(x):
    return np.isclose(x[0], L)

def upper_half(x):
    return np.logical_and(np.isclose(x[0], L), np.greater_equal(x[1], W/2))

def lower_half(x):
    return np.logical_and(np.isclose(x[0], L), np.less_equal(x[1], W/2))

def left_half(x):
    return np.logical_and(np.isclose(x[0], L), np.greater_equal(x[2], W/2))

def right_half(x):
    return np.logical_and(np.isclose(x[0], L), np.less_equal(x[2], W/2))
    

gdim = domain.topology.dim
fdim = gdim - 1

boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

#add tags to facets for the load surface

facet_tag = utils.set_meshtags(domain, fdim, load_surface, 1)
upper_half_tag = utils.set_meshtags(domain, fdim, upper_half, 2)
lower_half_tag = utils.set_meshtags(domain, fdim, lower_half, 3)
left_half_tag = utils.set_meshtags(domain, fdim, left_half, 4)
right_half_tag = utils.set_meshtags(domain, fdim, right_half, 5)


u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
alpha = fem.Constant(domain, default_scalar_type((0)))

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

tang_vect = utils.unit_tangent_vector(0, [0,W/2,W/2], gdim, 1.0)
space_varying_func = fem.Function(V)
space_varying_func.interpolate(tang_vect)
x = ufl.SpatialCoordinate(domain)
Mx_unit = fem.assemble_scalar(fem.form(ufl.cross(x, space_varying_func)[0] * ds(1)))
alpha = 1200/Mx_unit
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(alpha*space_varying_func, v) * ds(1)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

u_bc = fem.Function(V)

F1 = utils.print_forces(space_varying_func, domain, facet_tag, 1)
F2 = utils.print_forces(space_varying_func, domain, upper_half_tag, 2)
F3 = utils.print_forces(space_varying_func, domain, lower_half_tag, 3)
F4 = utils.print_forces(space_varying_func, domain, left_half_tag, 4)
F5 = utils.print_forces(space_varying_func, domain, right_half_tag, 5)


Mx = fem.assemble_scalar(fem.form(ufl.cross(x, alpha*space_varying_func)[0] * ds(1)))
print(f"Mx = {Mx}")


with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)

with io.XDMFFile(domain.comm, "vector_field.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "vfield"
    xdmf.write_function(space_varying_func)




# %%
