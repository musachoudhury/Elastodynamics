#This code is based off of Jeremy Bleyers Transient elastodynamics with Newmark time-integration tutorial
#https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, clear_output

from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType

L = 8.0
H = 0.2
B = 0.1
domain = create_box(
    MPI.COMM_WORLD,
    [[0.0, -B / 2, -H / 2], [L, B / 2, H / 2]],
    [8, 2, 2],
    CellType.hexahedron,
)

dim = domain.topology.dim
dx = ufl.Measure("dx", domain=domain)

degree = 2
shape = (dim,)
V = fem.functionspace(domain, ("Q", degree, shape))

u = fem.Function(V, name="Displacement")

def left(x):
    return np.isclose(x[0], 0.0)

def bottom(x):
    return np.isclose(x[1], 0.0)

# def top(x):
#     return np.isclose(x[1], 0.0)


def point(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)

# u, v, w
bottom_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, bottom)
u_bottom_dofs = fem.locate_dofs_topological(V.sub(0), domain.topology.dim - 1, bottom_facets)
w_bottom_dofs = fem.locate_dofs_topological(V.sub(2), domain.topology.dim - 1, bottom_facets)

#clamped_dofs = fem.locate_dofs_geometrical(V, left)
point_dof = fem.locate_dofs_geometrical(V, point)[0]
point_dofs = np.arange(point_dof * dim, (point_dof + 1) * dim)


#bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]
#bcs = [fem.dirichletbc(np.zeros((dim,)), u_bottom_dofs, V), fem.dirichletbc(np.zeros((dim,)), w_bottom_dofs, V)]
bcs = []
E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)
rho = fem.Constant(domain, 7.8e-3)
f = fem.Constant(domain, (0.0,) * dim)

f_max = 1.0
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

Nsteps = 200
Nsave = 100
times = np.linspace(0, 2, Nsteps + 1)
save_freq = Nsteps // Nsave
energies = np.zeros((Nsteps + 1, 3))
tip_displacement = np.zeros((Nsteps + 1, 3))
for i, dti in enumerate(np.diff(times)):
    if i % save_freq == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti

    T = np.array([1.0, 0.0, 0.0])
    if t <= 0.2:
        f.value = T * t / 0.2
    elif t>= 0.2: #and t<=0.4:
        f.value = - T * t / 0.2 + 2*T
    # else:
    #     f.value *= 0.0

    problem.solve()

    u.x.scatter_forward()  # updates ghost values for parallel computations

    # compute new velocity v_n+1
    v_new.interpolate(v_expr)

    # compute new acceleration a_n+1
    a_new.interpolate(a_expr)

    # update u_n with u_n+1
    #u0.x.array[:] = u.x.array

    u_old.x.array[:] = u.x.array
    #u.vector.copy(u_old.vector)

    # update v_n with v_n+1
    #v_new.vector.copy(v_old.vector)
    v_old.x.array[:] = v_new.x.array

    # update a_n with a_n+1
    
    #a_new.vector.copy(a_old.vector)
    a_old.x.array[:] = a_new.x.array

    energies[i + 1, 0] = fem.assemble_scalar(E_el)
    energies[i + 1, 1] = fem.assemble_scalar(E_kin)
    energies[i + 1, 2] = energies[i, 2] + dti * fem.assemble_scalar(P_damp)

    tip_displacement[i + 1, :] = u.x.array[point_dofs]

    clear_output(wait=True)
    print(f"Time increment {i+1}/{Nsteps}")

vtk.close()

cmap = plt.get_cmap("plasma")
colors = cmap(times / max(times))

I_y = B * H**3 / 12
omega_y = 1.875**2 * np.sqrt(float(E) * I_y / (float(rho) * B * H * L**4))
omega_z = omega_y * B / H
fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
lines = ax.plot(
    max(tip_displacement[:, 1]) * np.sin(omega_z * times),
    max(tip_displacement[:, 2]) * np.sin(omega_y * times),
    "--k",
    alpha=0.7,
)
ax.set_ylim(-2, 2)
ax.set_xlim(-3, 3)
markers = []


def draw_frame(n):
    markers.append(
        ax.plot(tip_displacement[n, 1], tip_displacement[n, 2], "o", color=colors[n])[0]
    )
    return markers


anim = animation.FuncAnimation(fig, draw_frame, frames=Nsteps, interval=20, blit=True)
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