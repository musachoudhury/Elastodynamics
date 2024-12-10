import numpy as np
from dolfinx import fem
import ufl

def left(x):
    return np.isclose(x[0], 0.0)

def bottom(x):
    return np.isclose(x[2], 0.0)

# def faces():
#     # if face=="left":
#     #     return np.isclose(x[0], 0.0)

#     # elif face=="bottom":
#     #     return np.isclose(x[2], 0.0)
#     return [left, bottom]

# def material_properties(domain, x):
#     E = fem.Constant(domain, x[0])
#     nu = fem.Constant(domain, x[1])
#     rho = fem.Constant(domain, x[2])

#     lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
#     mu = E / 2 / (1 + nu)
#     return E, nu, rho, lmbda, mu

# def forces(F, x):
#     area = x[0]*x[1]
#     return [F[0]/area, F[1]/area, F[2]/area]

class mechanics:
    # Constructor
    def __init__(self, domain, x):
        self.E = fem.Constant(domain, x[0])
        self.nu = fem.Constant(domain, x[1])
        self.rho = fem.Constant(domain, x[2])

        self.lmbda = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.E / 2 / (1 + self.nu)
        self.domain = domain
        self.dim = self.domain.topology.dim
        
        self.eta_M = fem.Constant(domain, 1e-2)
        self.eta_K = fem.Constant(domain, 1e-2)

    def material_properties(self):
        return self.E, self.nu, self.rho, self.lmbda, self.mu
    
    def forces(self, F, x):
        traction = fem.Constant(self.domain, (0.0,) * self.dim)

        area = x[0]*x[1]
        traction.value = np.array([F[0]/area, F[1]/area, F[2]/area])
        return traction
    
    def epsilon(self, v):
        return ufl.sym(ufl.grad(v))
    
    def sigma(self, v):
         return self.lmbda * ufl.tr(self.epsilon(v)) * ufl.Identity(self.dim) + 2 * self.mu * self.epsilon(v)

    def mass(self, u, u_):
        return self.rho * ufl.dot(u, u_) * ufl.dx
    
    def stiffness(self, u, u_):
        return ufl.inner(self.sigma(u), self.epsilon(u_)) * ufl.dx
    
    def damping(self, u, u_):
        return 0.0*self.eta_M * self.mass(u, u_) + 0.0*self.eta_K * self.stiffness(u, u_)





