#This code is based off of Jeremy Bleyers Transient elastodynamics with Newmark time-integration tutorial
#https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, clear_output

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.mesh import create_box, CellType
import utils
from math import sin, pi

a = 0.5
b = 0.5 
c = 6.0
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

def pressure_surface(x):

    height = np.less(x[2], c/12)
    sideOne = np.isclose(x[0], a/2)
    sideTwo = np.isclose(x[1], b/2)
    sideThree = np.isclose(x[0], -a/2)
    sideFour = np.isclose(x[1], -b/2)

    return height#np.logical_and(height, (sideOne or sideTwo or sideThree or sideFour))

pressure_surface_tags = utils.set_meshtags(domain, gdim-1, pressure_surface, 1)

clamped_dofs = fem.locate_dofs_geometrical(V, left)
# point_dof = fem.locate_dofs_geometrical(V, point)[0]
# point_dofs = np.arange(point_dof * dim, (point_dof + 1) * dim)

#bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]
bcs = []

E = fem.Constant(domain, 3.0e6)
nu = fem.Constant(domain, 0.499)
rho = fem.Constant(domain, 920.0)
f = fem.Constant(domain, (0.0,) * dim)
T = fem.Constant(domain, (0.0,) * dim)

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

# %%
class InitialCondition():
    def __init__(self, v0, h):
        self.v0 = v0
        self.h = h
        
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)

        values[0] = 0.0
        #self.v0*x[2]/self.h
        # for i in range(x.shape[1]):
        #     if abs(x[2, i] - 6.0) < 1e-6:
        #         values[0, i] = self.v0        
        return values

# class velocityFunction():
#     def __init__(self, v0, h):
#         self.v0 = v0
#         self.h = h
        
#     def __call__(self, x):
#         values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)


#         for i in range(x.shape[1]):
#             if abs(x[2, i] - 6.0) < 1e-6:
#                 values[0, i] = self.v0        
        
#         return values
    
v_old.interpolate(InitialCondition(10, 6))

#%%
u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

eta_M = fem.Constant(domain, 1e-4)
eta_K = fem.Constant(domain, 1e-4)


def mass(u, u_):
    return rho * ufl.dot(u, u_) * ufl.dx


def stiffness(u, u_):
    return ufl.inner(sigma(u), epsilon(u_)) * ufl.dx


def damping(u, u_):
    return eta_M * mass(u, u_) + eta_K * stiffness(u, u_)

ds = ufl.Measure("ds", domain=domain, subdomain_data=pressure_surface_tags)

Id = ufl.Identity(dim)

# Deformation gradient
F = ufl.variable(Id + ufl.grad(u))

# Right Cauchy-Green tensor
C = F.T * F

# Invariants of deformation tensors
I1 = ufl.tr(C)
J = ufl.det(F)
E_GL = ufl.variable(0.5*(C-Id))

#psi = mu / 2 * (I1 - 3 - 2 * ufl.ln(J)) + lmbda / 2 * (J - 1) ** 2

psi = lmbda/2 * ufl.tr(E_GL)**2 + mu * ufl.tr(E_GL*E_GL)

# PK1 stress = d_psi/d_F
P = ufl.diff(psi, F)

#stiffness(u, u_)
Residual = mass(a, u_) + damping(v, u_) + ufl.inner(P, ufl.grad(u_)) * ufl.dx - ufl.dot(f, u_) * ufl.dx - ufl.dot(T, u_) * ds(1)

#Residual_du = ufl.replace(Residual, {u: du})

#Jacobian = ufl.derivative(Residual_du, u, du)
# a_form = ufl.lhs(Residual_du)
# L_form = ufl.rhs(Residual_du)

# problem = fem.petsc.LinearProblem(
#     a_form, L_form, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
# )

problem = fem.petsc.NonlinearProblem(Residual, u, bcs)

E_kin = fem.form(0.5 * mass(v_old, v_old))
E_el = fem.form(0.5 * stiffness(u_old, u_old))
P_damp = fem.form(damping(v_old, v_old))

x = ufl.SpatialCoordinate(domain)
A_mom1 = fem.form(rho*ufl.cross((x+u_old), v_old)[0]*ufl.dx)
A_mom2 = fem.form(rho*ufl.cross((x+u_old), v_old)[1]*ufl.dx)
A_mom3 = fem.form(rho*ufl.cross((x+u_old), v_old)[2]*ufl.dx)

vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")

t = 0.0

Nsteps = 100
Nsave = 100
times = np.linspace(0, 10, Nsteps + 1)
save_freq = Nsteps // Nsave

energies = np.zeros((Nsteps + 1, 3))
angular_mom = np.zeros((Nsteps + 1, 3))
tip_displacement = np.zeros((Nsteps + 1, 3))

solver = NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

for i, dti in enumerate(np.diff(times)):
    if i % save_freq == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti

    if t <= 0.2:
        f.value = 0.0#np.array([0.0, 1.0, 1.5]) * t / 0.2
    else:
        f.value *= 0.0
    
    # if t <= 0.5:
    #     T.value = np.array([1.0, 0.0, 0.0])*t
    # elif t > 0.5:
    #     T.value = -np.array([1.0, 0.0, 0.0]) * t + 1.0
    # else:
    #     T.value = 0.0
    if t <= 1.0:
        T.value = 1e3*sin(4*pi*t)*np.array([1.0, 0.0, 0.0])
    else:
        T.value = 0.0

    #problem.solve()
    num_its, converged = solver.solve(u)

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

    angular_mom[i + 1, 0] = fem.assemble_scalar(A_mom1)
    angular_mom[i + 1, 1] = fem.assemble_scalar(A_mom2)
    angular_mom[i + 1, 2] = fem.assemble_scalar(A_mom3)

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

plt.figure(1)
plt.plot(times, energies[:, 0], label="Elastic")
plt.plot(times, energies[:, 1], label="Kinetic")
plt.plot(times, energies[:, 2], label="Damping")
plt.plot(times, np.sum(energies, axis=1), label="Total")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energies")
plt.show()

plt.figure(2)
plt.plot(times, angular_mom[:, 0], label="Component 1")
plt.plot(times, angular_mom[:, 1], label="Component 2")
plt.plot(times, angular_mom[:, 2], label="Component 3")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Angular Momentum")
plt.show()
# %%
