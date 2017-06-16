"""Standalone script for demonstrating how to work with rigid motions
in DOLFIN and linear elasticity. Based on
dolfin/demo/undocumented/elasticity/.

Tested with FEniCS 2017.1.0dev
"""

# Author: Marie E. Rognes, June 16 2017

from __future__ import print_function
from dolfin import *

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
parameters["linear_algebra_backend"] = "PETSc"

def build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

# Define mesh
n = 4
mesh = UnitCubeMesh(n, n, n)

#  Define some loading
f = Expression(("(x[0] - 0.5) + 0.1", "(x[1] - 0.5)", "0.0"), degree=1)

# Elasticity parameters
mu = Constant(100.0)
lmbda = Constant(1000.0)

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*div(v)*Identity(len(v))

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx
L = inner(f, v)*dx

# Assemble system
A = assemble(a)
b = Vector(mpi_comm_world(), V.dim())
assemble(L, tensor=b)

# Create solution function
u = Function(V)

# Create null space basis
null_space = build_nullspace(V, u.vector())

# Associate null space with A
as_backend_type(A).set_nullspace(null_space)
as_backend_type(A).set_near_nullspace(null_space)


# Orthogonalize right-hand side to make sure that input is in the
# range of A aka the orthogonal complement of the null space, cf.
# linear algebra 101.
b0 = b.copy()
null_space.orthogonalize(b)

# Print difference between orthogonalized and non-orthogonalized
# right-hand side b 
b0.axpy(-1, b)
print(b0.norm("l2"))

# Define solver, set operator and set nullspace
solver = PETScKrylovSolver("cg", "hypre_amg")
solver.parameters["monitor_convergence"] = True
solver.set_operator(A)

# Compute solution
solver.solve(u.vector(), b);

# Plot solution
plot(u, interactive=True)
