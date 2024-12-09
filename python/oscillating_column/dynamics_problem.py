#This code is based off of Jeremy Bleyers Transient elastodynamics with Newmark time-integration tutorial
#https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, clear_output

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType

a = 0.5
b = 0.5 
c = 6
domain = create_box(
    MPI.COMM_WORLD,
    [[-a/2, -b/2, 0.0], [a/2, b/2, c]],
    [2, 2, 8],
    CellType.hexahedron,
)

dim = domain.topology.dim
gdim = dim
dx = ufl.Measure("dx", domain=domain)

degree = 2
shape = (dim,)
V = fem.functionspace(domain, ("Q", degree, shape))

u = fem.Function(V, name="Displacement")

def left(x):
    return np.isclose(x[2], 0.0)


def point(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)


clamped_dofs = fem.locate_dofs_geometrical(V, left)
# point_dof = fem.locate_dofs_geometrical(V, point)[0]
# point_dofs = np.arange(point_dof * dim, (point_dof + 1) * dim)


bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]

E = fem.Constant(domain, 17.0e6)
nu = fem.Constant(domain, 0.3)
rho = fem.Constant(domain, 1100.0)
f = fem.Constant(domain, (0.0,) * dim)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)

u_old = fem.Function(V)
v_old = fem.Function(V)
a_old = fem.Function(V)
a_new = fem.Function(V)
v_new = fem.Function(V)

beta_ = 0.25
beta = fem.Constant(domain, beta_)
gamma_ = 0.5
gamma = fem.Constant(domain, gamma_)
dt = fem.Constant(domain, 0.0)

a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
a_expr = fem.Expression(a, V.element.interpolation_points())

v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
v_expr = fem.Expression(v, V.element.interpolation_points())

class InitialCondition():
    def __init__(self, v0, h):
        self.v0 = v0
        self.h = h
        
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = self.v0*x[2]/self.h
        return values

v_old.interpolate(InitialCondition(10, 6))

u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

eta_M = fem.Constant(domain, 1e-2)
eta_K = fem.Constant(domain, 1e-2)


def mass(u, u_):
    return rho * ufl.dot(u, u_) * ufl.dx


def stiffness(u, u_):
    return ufl.inner(sigma(u), epsilon(u_)) * ufl.dx


def damping(u, u_):
    return eta_M * mass(u, u_) + eta_K * stiffness(u, u_)


Residual = mass(a, u_) + damping(v, u_) + stiffness(u, u_) - ufl.dot(f, u_) * ufl.dx

Residual_du = ufl.replace(Residual, {u: du})
a_form = ufl.lhs(Residual_du)
L_form = ufl.rhs(Residual_du)

problem = fem.petsc.LinearProblem(
    a_form, L_form, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

E_kin = fem.form(0.5 * mass(v_old, v_old))
E_el = fem.form(0.5 * stiffness(u_old, u_old))
P_damp = fem.form(damping(v_old, v_old))

vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")

t = 0.0

Nsteps = 100
Nsave = 100
times = np.linspace(0, 8, Nsteps + 1)
save_freq = Nsteps // Nsave

energies = np.zeros((Nsteps + 1, 3))
tip_displacement = np.zeros((Nsteps + 1, 3))
for i, dti in enumerate(np.diff(times)):
    if i % save_freq == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti

    if t <= 0.2:
        f.value = 0.0#np.array([0.0, 1.0, 1.5]) * t / 0.2
    else:
        f.value *= 0.0

    problem.solve()

    u.x.scatter_forward()  # updates ghost values for parallel computations

    # compute new velocity v_n+1
    v_new.interpolate(v_expr)

    # compute new acceleration a_n+1
    a_new.interpolate(a_expr)

    u_old.x.array[:] = u.x.array

    # update v_n with v_n+1
    v_old.x.array[:] = v_new.x.array

    # update a_n with a_n+1
    
    a_old.x.array[:] = a_new.x.array

    energies[i + 1, 0] = fem.assemble_scalar(E_el)
    energies[i + 1, 1] = fem.assemble_scalar(E_kin)
    energies[i + 1, 2] = energies[i, 2] + dti * fem.assemble_scalar(P_damp)

    #tip_displacement[i + 1, :] = u.x.array[point_dofs]

    clear_output(wait=True)
    print(f"Time increment {i+1}/{Nsteps}")

vtk.close()

# cmap = plt.get_cmap("plasma")
# colors = cmap(times / max(times))

# I_y = B * H**3 / 12
# omega_y = 1.875**2 * np.sqrt(float(E) * I_y / (float(rho) * B * H * L**4))
# omega_z = omega_y * B / H
# fig = plt.figure()
# ax = fig.gca()
# ax.set_aspect("equal")
# lines = ax.plot(
#     max(tip_displacement[:, 1]) * np.sin(omega_z * times),
#     max(tip_displacement[:, 2]) * np.sin(omega_y * times),
#     "--k",
#     alpha=0.7,
# )
# ax.set_ylim(-2, 2)
# ax.set_xlim(-3, 3)
# markers = []


# def draw_frame(n):
#     markers.append(
#         ax.plot(tip_displacement[n, 1], tip_displacement[n, 2], "o", color=colors[n])[0]
#     )
#     return markers


#anim = animation.FuncAnimation(fig, draw_frame, frames=Nsteps, interval=20, blit=True)
plt.close()
#HTML(anim.to_html5_video())

plt.plot(times, energies[:, 0], label="Elastic")
plt.plot(times, energies[:, 1], label="Kinetic")
plt.plot(times, energies[:, 2], label="Damping")
plt.plot(times, np.sum(energies, axis=1), label="Total")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energies")
plt.show()