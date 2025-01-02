# %%
#This code is based off of Jeremy Bleyers Transient elastodynamics with Newmark time-integration tutorial
#https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io, mesh
import gmsh
from dolfinx.io.gmshio import read_from_msh
import dolfinx.fem.petsc
import utils
import definitions as defs
gdim = 3  # domain geometry dimension
fdim = gdim-1  # facets dimension
w = 6
l = 3
h = 10

w2 = 3
h2 = 3

mesh_comm = MPI.COMM_WORLD
domain, cell_tags, facet_tags = read_from_msh('Lshapemesh.msh', mesh_comm, 0, gdim=gdim)

# %%
dim = domain.topology.dim
dx = ufl.Measure("dx", domain=domain)

degree = 1
shape = (dim,)

V = fem.functionspace(domain, ("Lagrange", degree, shape))

u = fem.Function(V, name="Displacement")

# %%

#bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]
bcs = []
mechanics = defs.mechanics(domain, [210e3, 0.3, 785.0])

E, nu, rho, lmbda, mu = mechanics.material_properties()

f = fem.Constant(domain, (0.0,) * dim)
P = fem.Constant(domain, (0.0))

# Loads on face A, facet tag 3
tractionA = mechanics.forces([1600, 800, 1600], [w2, l])
Ma_z = -1200

# Loads on face B, facet tag 8
tractionB = mechanics.forces([800, -1600, -800], [h2, l])
Mb_x = -1200
# %%
u_old = fem.Function(V)
v_old, v_new = utils.Functions(V)
a_old, a_new = utils.Functions(V)

beta_ = 0.25
beta = fem.Constant(domain, beta_)
gamma_ = 0.5
gamma = fem.Constant(domain, gamma_)
dt = fem.Constant(domain, 0.0)

a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
a_expr = fem.Expression(a, V.element.interpolation_points())

v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
v_expr = fem.Expression(v, V.element.interpolation_points())

u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
MA_Z = utils.moment_traction(Ma_z, 2, [w2/2 , l/2, h], domain, gdim, V, facet_tags , 3)
MB_X = utils.moment_traction(Mb_x, 0, [w , l/2, h2/2], domain, gdim, V, facet_tags, 8)
loadA = ufl.dot((tractionA+MA_Z)*P, u_) * ds(3)
loadB = ufl.dot((tractionB+MB_X)*P, u_) * ds(8)
Residual = mechanics.mass(a, u_) + mechanics.damping(v, u_) + mechanics.stiffness(u, u_) - ufl.dot(f, u_) * ufl.dx - loadA - loadB

#%%
Residual_du = ufl.replace(Residual, {u: du})
a_form = ufl.lhs(Residual_du)
L_form = ufl.rhs(Residual_du)

problem = fem.petsc.LinearProblem(
    a_form, L_form, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

E_kin = fem.form(0.5 * mechanics.mass(v_old, v_old))
E_el = fem.form(0.5 * mechanics.stiffness(u_old, u_old))
P_damp = fem.form(mechanics.damping(v_old, v_old))

x = ufl.SpatialCoordinate(domain)
A_mom1 = fem.form(rho*ufl.cross((x+u_old), v_old)[0]*ufl.dx)
A_mom2 = fem.form(rho*ufl.cross((x+u_old), v_old)[1]*ufl.dx)
A_mom3 = fem.form(rho*ufl.cross((x+u_old), v_old)[2]*ufl.dx)

vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")

t = 0.0

Nsteps = 1000
#Nsave = 1000
times = np.linspace(0, 100, Nsteps + 1)
#save_freq = Nsteps // Nsave
energies = np.zeros((Nsteps + 1, 3))
angular_mom = np.zeros((Nsteps + 1, 3))
tip_displacement = np.zeros((Nsteps + 1, 3))

f.value = np.array([0.0, 0.0, 0.0])

for i, dti in enumerate(np.diff(times)):
    if i % 10 == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti
    
    if t <= 0.5:
        P.value = t
    elif 0.5>=t and t<=1.0:
        P.value = 1.0 - t
    else:
        P.value = 0.0
    print(t)

    problem.solve()

    u.x.scatter_forward()  # updates ghost values for parallel computations

    # compute new velocity v_n+1
    v_new.interpolate(v_expr)

    # compute new acceleration a_n+1
    a_new.interpolate(a_expr)

    # update u_n with u_n+1

    u_old.x.array[:] = u.x.array

    # update v_n with v_n+1
    v_old.x.array[:] = v_new.x.array

    # update a_n with a_n+1
    
    a_old.x.array[:] = a_new.x.array

    energies[i + 1, 0] = fem.assemble_scalar(E_el)
    energies[i + 1, 1] = fem.assemble_scalar(E_kin)
    energies[i + 1, 2] = energies[i, 2] + dti * fem.assemble_scalar(P_damp)
    angular_mom[i + 1, 0] = fem.assemble_scalar(A_mom1)
    angular_mom[i + 1, 1] = fem.assemble_scalar(A_mom2)
    angular_mom[i + 1, 2] = fem.assemble_scalar(A_mom3)

    clear_output(wait=True)
    print(f"Time: {t}, Time increment {i+1}/{Nsteps}")

vtk.close()

cmap = plt.get_cmap("plasma")
colors = cmap(times / max(times))

plt.figure(1)
plt.plot(times, energies[:, 0], label="Elastic")
plt.plot(times, energies[:, 1], label="Kinetic")
plt.plot(times, energies[:, 2], label="Damping")
plt.plot(times, np.sum(energies, axis=1), label="Total")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energies")

plt.figure(2)
plt.plot(times, angular_mom[:, 0], label="Component 1")
plt.plot(times, angular_mom[:, 1], label="Component 2")
plt.plot(times, angular_mom[:, 2], label="Component 3")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Angular Momentum")
plt.show()
# %%
