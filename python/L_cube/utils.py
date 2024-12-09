from petsc4py import PETSc
from dolfinx import mesh, fem
import numpy as np
import ufl

class unit_tangent_vector():
    def __init__(self, axis, origin, gdim, alpha):
        self.axis = axis
        self.origin = origin
        self.alpha = alpha
        self.gdim = gdim
        
    def __call__(self, x):
        x_0 = x[0]-self.origin[0]
        x_1 = x[1]-self.origin[1]
        x_2 = x[2]-self.origin[2]
        if self.axis == 0:
            mag = 1.0/self.alpha#self.alpha*((x_1**2+(x_2)**2)**0.5)
            values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 0.0
            values[1] = 1/mag*(x_2)
            values[2] = 1/mag*-x_1
        elif self.axis == 1:
            mag = 1.0/self.alpha#(((x_0)**2+(x_2)**2)**0.5)
            values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 1/mag*(x_2)
            values[1] = 0.0
            values[2] = 1/mag*-(x_0)
        elif self.axis == 2:
            mag = 1.0/self.alpha#(((x_0)**2+x_1**2)**0.5)
            values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 1/mag*x_1
            values[1] = 1/mag*-(x_0)
            values[2] = 0.0
        return values

def moment_traction(M, axis, centre, domain, gdim, V, facet_tags, tag):
    tang_vect = unit_tangent_vector(axis, centre, gdim, 1.0)
    space_varying_func = fem.Function(V)
    space_varying_func.interpolate(tang_vect)
    x = ufl.SpatialCoordinate(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    M_unit = fem.assemble_scalar(fem.form(ufl.cross(x, space_varying_func)[axis] * ds(tag)))
    alpha = M/M_unit
    return alpha*space_varying_func


def set_meshtags(domain, fdim, bc, tag):
    facets = mesh.locate_entities_boundary(domain, fdim, bc)
    marked_facets = np.hstack([facets])
    marked_values = np.hstack([np.full_like(facets, tag)])
    sorted_facets = np.argsort(marked_facets)
    return mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])


def print_forces(traction, domain, tags, id):
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    print("**************************************")
    F_x = fem.assemble_scalar(fem.form(traction[0] * ds(id)))
    print(f"Fx = {F_x:.6f}")

    F_y = fem.assemble_scalar(fem.form(traction[1] * ds(id)))
    print(f"Fy = {F_y:.6f}")

    F_z = fem.assemble_scalar(fem.form(traction[2] * ds(id)))
    print(f"Fz = {F_z:.6f}")
    print("**************************************")

    return [F_x, F_y, F_z]