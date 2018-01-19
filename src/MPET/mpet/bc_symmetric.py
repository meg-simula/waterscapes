# Set boundary conditions and eliminate excess dofs
from numpy import array
from dolfin import *
def get_bc_dofs(bc):
    from numpy import intc
    return array(list(bc.get_boundary_values().keys()), dtype = intc)

def zero_rows_cols(dofs, A, b = None):
    # import pdb; pdb.set_trace()  
    # from IPython import embed; embed()  
    amat = as_backend_type(A).mat()
    if b != None:
        bvec = as_backend_type(b).vec()
    else:
        bvec = None
    amat.zeroRowsColumns(dofs, x = bvec, b = bvec)

def apply_symmetric(bc, A, b = None):
    bc_dofs = get_bc_dofs(bc)
    zero_rows_cols(bc_dofs, A, b)
