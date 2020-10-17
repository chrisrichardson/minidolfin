# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import numpy
from matplotlib import pyplot, tri
import scipy.sparse.linalg

import timeit
import math
import argparse

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import interpolate_vertex_values
from minidolfin.assembling import assemble
from minidolfin.bcs import build_dirichlet_dofs
from minidolfin.bcs import bc_apply


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="minidolfin Helmholtz demo",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--mesh-size", type=int, dest="n", default=128,
                    help="mesh resolution")
parser.add_argument("-f", action="append", dest="form_compiler_parameters",
                    metavar="parameter=value", default=[],
                    help="additional form compiler paramter")
parser.add_argument("-d", "--debug", action='store_true', default=False,
                    help="enable debug output")
args = parser.parse_args()


# Build form compiler parameters
form_compiler_parameters = {}
for p in args.form_compiler_parameters:
    k, v = p.split("=")
    form_compiler_parameters[k] = v

# Plane wave
omega2 = 15**2 + 11**2


def u_exact(x):
    return math.cos(-15*x[0] + 12*x[1])


# UFL form
element = ufl.FiniteElement("P", ufl.triangle, 3)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - omega2*ufl.dot(u, v))*ufl.dx

# Build mesh
mesh = build_unit_square_mesh(args.n, args.n)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

# Run and time assembly
t = -timeit.default_timer()
A = assemble(dofmap, a, dtype=numpy.float32)
t += timeit.default_timer()
print('Assembly time a: {}'.format(t))

# Prepare solution and rhs vectors and apply boundary conditions
x = numpy.zeros(A.shape[1], dtype=A.dtype)
b = numpy.zeros(A.shape[0], dtype=A.dtype)

# Set Dirichlet BCs

t = -timeit.default_timer()
bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_exact, dtype=A.dtype)

bc_apply(bc_dofs, bc_vals, A, b)

t += timeit.default_timer()
print('Apply BCs: {}'.format(t))

# Solve linear system
t = -timeit.default_timer()
x = scipy.sparse.linalg.spsolve(A, b)
r = (A*x - b)
print(r.max(), r.min())

t += timeit.default_timer()
print('Solve linear system time: {}'.format(t))

# Plot solution
vertex_values = interpolate_vertex_values(dofmap, x)
triang = tri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1],
                           mesh.get_connectivity(tdim, 0))
pyplot.tripcolor(triang, vertex_values)
pyplot.show()
