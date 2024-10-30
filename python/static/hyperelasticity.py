#This code is based off of Jeremy Bleyers Hyperelasticity tutorial
#https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html

import numpy as np
import ufl
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import fem, io, nls
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
)

L = 3.0
N = 4
mesh = create_box(
    MPI.COMM_WORLD,
    [[-0.5, -0.5, 0.0], [0.5, 0.5, L]],
    [N, N, 4 * N],
    CellType.hexahedron,
)

dim = mesh.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)
V = fem.functionspace(mesh, ("P", degree, shape))

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
mu = fem.Constant(mesh, E / 2 / (1 + nu))
lmbda = fem.Constant(mesh, E * nu / (1 - 2 * nu) / (1 + nu))

# Stored strain energy density (compressible neo-Hookean model)
psi = mu / 2 * (I1 - 3 - 2 * ln(J)) + lmbda / 2 * (J - 1) ** 2

# Stored strain energy density (St. Venant-Kirchhoff model)
#psi = lmbda/2 * tr(E_GL)**2 + mu * tr(E_GL*E_GL)

# PK1 stress = d_psi/d_F
P = ufl.diff(psi, F)
#print(P)

def bottom(x):
    return np.isclose(x[2], 0.0)


def top(x):
    return np.isclose(x[2], L)


bottom_dofs = fem.locate_dofs_geometrical(V, bottom)
top_dofs = fem.locate_dofs_geometrical(V, top)

u_bot = fem.Function(V)
u_top = fem.Function(V)

bcs = [fem.dirichletbc(u_bot, bottom_dofs), fem.dirichletbc(u_top, top_dofs)]

x = SpatialCoordinate(mesh)
theta = fem.Constant(mesh, 0.0)
Rot = as_matrix([[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]])
print(Rot)
rotation_displ = dot(Rot, x) - x
rot_expr = fem.Expression(rotation_displ, V.element.interpolation_points())

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})
E_pot = psi * dx

v = TestFunction(V)
du = TrialFunction(V)
# Residual = derivative(
#     E_pot, u, v
# )  # This is equivalent to Residual = inner(P, grad(v))*dx

Residual = inner(P, grad(v))*dx
Jacobian = derivative(Residual, u, du)

problem = fem.petsc.NonlinearProblem(Residual, u, bcs)

solver = nls.petsc.NewtonSolver(mesh.comm, problem)
# Set Newton solver options
solver.atol = 1e-4
solver.rtol = 1e-4
solver.convergence_criterion = "incremental"

angle_max = 2 * np.pi
Nsteps = 30

out_file = "hyperelasticity.xdmf"
with io.XDMFFile(mesh.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(mesh)

u.vector.set(0.0)

def sigma(v):
    return lmbda * ufl.tr(grad(v)) * ufl.Identity(dim) + 2 * mu * grad(v)


def stiffness(u, u_):
    return inner(1/J*F*P, grad(u_))*dx

E_el = fem.form(0.5 * stiffness(u, u))

energies = np.zeros((Nsteps + 1, 1))
#E_el = fem.form(0.5 * Residual)

angle_steps = np.linspace(0, angle_max, int(Nsteps/2) + 1)[1:]
angle_steps = np.concatenate((angle_steps, np.flip(angle_steps)))
print(angle_steps)


for n, angle in enumerate(angle_steps):
    theta.value = angle
    u_top.interpolate(rot_expr)
    
    num_its, converged = solver.solve(u)
    assert converged

    u.x.scatter_forward()  # updates ghost values for parallel computations

    energies[n + 1, 0] = fem.assemble_scalar(E_el)

    print(
        f"Time step {n}, Number of iterations {num_its}, Angle {angle*180/np.pi:.0f} deg."
    )

    with io.XDMFFile(mesh.comm, out_file, "a") as xdmf:
        xdmf.write_function(u, n + 1)


times = np.linspace(0, 2, Nsteps + 1)
plt.plot(times, energies[:, 0], label="Elastic")
plt.plot(times, np.sum(energies, axis=1), label="Total")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energies")
plt.show()