# Set boundary conditions in a symmetric fashion and eliminate excess dofs
from numpy import array
from dolfin import *

#getting the dofs associated to DirichletBC
def get_bc_dofs(bc):
    from numpy import intc
    return array(list(bc.get_boundary_values().keys()), dtype = intc)

#Eliminate rows and colums associated to Dirichlet dofs and modifies rhs accordingly
def zero_rows_cols(dofs, A, b = None):
    amat = as_backend_type(A).mat()

    if b is None:
        bvec = None
    else:
        bvec = as_backend_type(b).vec()
    amat.zeroRowsColumns(dofs, x = bvec, b = bvec)

def apply_symmetric(bc, A, b = None):
    bc_dofs = get_bc_dofs(bc)
    zero_rows_cols(bc_dofs, A, b)
