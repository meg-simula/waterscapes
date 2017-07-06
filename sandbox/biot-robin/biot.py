# Biot's equations preconditioned solver (ignoring acceleration): find the displacement u
# and the pore/fluid pressure p such that:
#
#             - div(sigma(u) - alpha p I) = f
#  s_0 p_t + alpha div u_t - div k grad p = g
# 
# Backward Euler in time gives for each time-step n = 1, 2, ...: 
#  
#               - div(sigma(u^n) - alpha p^n I) = f^n
# s_0 p^n + alpha div u^n  - dt div(k grad p^n) = dt g^n - s_0 p^{n-1} + alpha div u^{n-1}
#
# Denote u = u^n, p = p^n and p^{n-1} = p0, u^{n-1} = u0 for brevity
#
# Standard two-field formulation and integration by parts:
#
# < sigma(u) - alpha p I, eps(v) > - << sigma(u)*n - alpha p n, v >> = <f^n, v>
# < s_0 p + alpha div u, q > + < dt k grad p, grad q > - << dt k grad p * n, q >> = < dt g^n - s_0 p0 + alpha div u0, q>
#
# where <.,.> denotes the L^2 inner product over the domain and <<.,.>> over the boundary.
#
# NOTE: The system is nondimensionalised. The pressure solution obtained is actually the pressure difference
# given by p = p_brain - delta_p, where delta_p is the pressure difference between the ventricle and subaracnoidal CSF pressures.
#
#### Description adapted from existing description in the waterscapes repository

# Matteo Croci, July 6 2017

from dolfin import *
from petsc4py import PETSc
from numpy import hstack, unique, array, searchsorted

# Setting C++ optimisation flags
parameters['form_compiler']['optimize']=True
parameters['form_compiler']['cpp_optimize']=True
parameters['form_compiler']['representation']='uflacs'
parameters['form_compiler']['cpp_optimize_flags']='-O3 -ffast-math -march=native'

dolfin.SubSystemsManager.init_petsc()

# setting this flag to False activates LU which is faster
# setting this flag to True activates the precondtioner
# the preconditioner is slower than LU (by how much depends on the atol set)
fieldsplit_on = False

# NOTE: this preconditioner is ad hoc for this problem, but not much robust to
# parameter variation. Generally changing w_ksp_bcgsl_ell and the number of
# multigrid cycles in w_fieldsplit_1_ does the job.

# outer krylov solver : enhanced BiCGStab
# outer preconditioner: fieldsplit block Gauss-Seidel (multiplicative) for displacement and pressure.
# displacement solver : split the components (multiplicative fieldsplit again)
#                       and use 1 hypre algebraic multigrid (AMG) V-cycle as preconditioner for both components.
# pressure solver     : apply multiple AMG cycles

opts   = {"w_ksp_type" : "bcgsl",
          "w_ksp_bcgsl_ell" : 4, # if solver stagnates, decrease this number
          #"w_ksp_gmres_restart" : 115,   # this is useful if fgmres is selected
          #"w_ksp_gmres_preallocate" : None,
          #"w_ksp_gmres_modifypcnochange" : None,
          "w_ksp_atol" : 1.0e-9,
          "w_ksp_rtol" : 1.0e-11,
          "w_ksp_max_it" : 1000,
          "w_pc_factor_mat_solver_package" : "mumps",
          "w_ksp_converged_reason" : None,
          "w_ksp_monitor_true_residual" : None,

          "w_pc_type" : "fieldsplit",
          "w_pc_fieldsplit_type" : "multiplicative",
          "w_pc_fieldsplit_0_fields" : 0,
          "w_pc_fieldsplit_1_fields" : 1,

          "w_fieldsplit_0_ksp_type" : "preonly",
          "w_fieldsplit_0_ksp_atol" : 1.0e-1,
          "w_fieldsplit_0_ksp_rtol" : 1.0e-15,
          "w_fieldsplit_0_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_0_ksp_converged_reasonx" : None,
          "w_fieldsplit_0_pc_factor_mat_solver_package" : "mumps",
          "w_fieldsplit_0_ksp_max_it" : 1,

          "w_fieldsplit_0_pc_type" : "fieldsplit",
          "w_fieldsplit_0_pc_fieldsplit_type" : "multiplicative",
          "w_fieldsplit_0_pc_fieldsplit_0_fields" : 0,
          "w_fieldsplit_0_pc_fieldsplit_1_fields" : 1,

          "w_fieldsplit_0_fieldsplit_ksp_type" : "preonly",
          "w_fieldsplit_0_fieldsplit_ksp_atol" : 1.0e-10,
          "w_fieldsplit_0_fieldsplit_ksp_rtol" : 1.0e-15,
          "w_fieldsplit_0_fieldsplit_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_0_fieldsplit_ksp_converged_reasonx" : None,
          "w_fieldsplit_0_fieldsplit_pc_factor_mat_solver_package" : "mumps",
          "w_fieldsplit_0_fieldsplit_ksp_max_it" : 1,
          "w_fieldsplit_0_fieldsplit_pc_type" : "hypre",
          "w_fieldsplit_0_fieldsplit_pc_hypre_type" : "boomeramg",
          ## hypre boomeramg optimisation flags
          "w_fieldsplit_0_fieldsplit_pc_hypre_boomeramg_nodal_coarsen" : 6,
          "w_fieldsplit_0_fieldsplit_pc_hypre_boomeramg_vec_interp_variant" : 3,

          "w_fieldsplit_1_ksp_type" : "richardson",
          "w_fieldsplit_1_ksp_atol" : 1.0e-10,
          "w_fieldsplit_1_ksp_rtol" : 1.0e-15,
          "w_fieldsplit_1_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_1_ksp_converged_reasonx" : None,
          "w_fieldsplit_1_pc_factor_mat_solver_package" : "mumps",
          "w_fieldsplit_1_ksp_max_it" : 3,
          "w_fieldsplit_1_pc_type" : "hypre",
          "w_fieldsplit_1_pc_hypre_type" : "boomeramg",
          ## hypre boomeramg optimisation flags
          "w_fieldsplit_1_pc_hypre_boomeramg_nodal_coarsen" : 1,
          "w_fieldsplit_1_pc_hypre_boomeramg_vec_interp_variant" : 1,
          }

# if we use LU, change the options to select LU only
if fieldsplit_on is False:
    opts["w_ksp_type"] = "preonly"
    opts["w_pc_type"] = "lu"
    opts.pop("w_ksp_converged_reason", None)

# set PETSc options
petsc_opts = PETSc.Options()
for key in opts:
    petsc_opts[key] = opts[key]

def extract_sub_matrix(A, subspace_in, subspace_out):
    Amat  = as_backend_type(A).mat()

    subis_in   = PETSc.IS()
    subdofs_in = W.sub(subspace_in).dofmap().dofs()
    subis_in.createGeneral(subdofs_in)

    subis_out   = PETSc.IS()
    subdofs_out = W.sub(subspace_out).dofmap().dofs()
    subis_out.createGeneral(subdofs_out)

    submat  = Amat.getSubMatrix(subis_out, subis_in)

    return submat

def extract_sub_vector(V, subspace):
    #Vvec  = as_backend_type(V.vector()).vec()
    Vvec  = as_backend_type(V).vec()

    subis   = PETSc.IS()
    subdofs = W.sub(subspace).dofmap().dofs()
    subis.createGeneral(subdofs)

    subvec  = Vvec.getSubVector(subis)
    dupe    = subvec.copy()
    Vvec.restoreSubVector(subis, subvec)

    return dupe

# load meshes. The meshes must be generated by running the 2D_generate_mesh.py script
mesh = Mesh()
file = HDF5File(mpi_comm_world(), "2D_circle_brain.h5", "r")
file.read(mesh, "/inner_mesh", False)
#outer_mesh = Mesh() # NOTE outer_mesh is not used for the moment
#file.read(outer_mesh, "/outer_mesh", False)
file.close()

# domain volume, needed later for output functionals
vol = assemble(Constant(1.0)*dx(domain = mesh))

# load domain markers. These are generated with the 2D_generate_mesh.py script
markers = MeshFunction("size_t", mesh, "markers.xml")
DG = FunctionSpace(mesh, "DG", 0)
Chi = Function(DG)
# characteristic function of the grey matter regions
Chi.vector()[:] = markers.array() 

# load boundary IDs. These are generated with the 2D_generate_mesh.py script
# boundary IDs: 1- ventricles, 2- SAS, 3- aqueduct
boundary_markers = MeshFunction("size_t",  mesh, "boundary_markers.xml")
File("boundary_markers.pvd") << boundary_markers
ds = ds(subdomain_data = boundary_markers)

# Define other material parameters. The length unit is cm.
#E = 584.*10 # conversion_factor(from Pa to g/cm/s^2) = 10
# mu = Constant(E/(2.*(1+nu))) # conversion from E
nu = 0.479 # Marie dixit
mu = 5000.*10. # Marie dixit [conversion factor from Pa to g/cm/s^2 is 10]
lmbda = 2.*mu*nu/(1.-2.*nu) # lame' parameters
# Define permeabilities and divide them by the ISF viscosity (similar as for water). Values taken from Shahim
# this is m^2*[conversion factor to cm^2]/([ISF viscosity in g/(mm*s)]/[conversion factor from mm to cm])
Kgrey_val = 1.e-15*1.0e4/(0.658e-3/10.)*2.0e1 # permeabilities. There is a factor of 20 that is added to make results look nicer :)
Kpara_val = 1.e-13*1.0e4/(0.658e-3/10.)*2.0e1
pin_val = 0.15*10.*133.32 # CSF pressure
# NOTE: the permeability values are unknown
beta_pia_val = 1e-6*10 # membrane permeability at the pia interface, units needed are g/(cm^2*s)
beta_ventricle_val = 1e-13*10 # membrane permeability at the ventricular interface

# scale factors for nondimensionalisation
xscale = 1.0
Kscale = xscale**2/lmbda
p_scale = pin_val
u_scale = pin_val/lmbda*xscale

# scale the brain permeabilities
Kgrey = Constant(Kgrey_val/Kscale)
Kpara = Constant(Kpara_val/Kscale)
K = (Chi*Kgrey + (Constant(1.0) - Chi)*Kpara)

# time step
dt = Constant(0.01)

print "nondimentionalised input parameters, (2*mu/lmbda*xscale, Kwhite, beta): (%f, %f, %f)" % (2.0*mu/lmbda*xscale, Kpara_val/Kscale, beta_pia_val/Kscale)

# Volume force/source
f = Constant((0.0, 0.0))
g = Constant(0.0)

n = FacetNormal(mesh)

# Define finite element spaces
Velem = VectorElement("CG", mesh.ufl_cell(), 2)
Qelem = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, Velem*Qelem)
U = FunctionSpace(mesh, Velem)
V = FunctionSpace(mesh, Qelem)

# totally not confusing one-liner that extract the y-coordinates of all the mesh nodes on the aqueduct
# we need this to compute the actual position of the aqueduct corners to interpolate the bcs
aqueductY = array([Vertex(mesh, item).point().y() for item in unique(hstack([facet.entities(0) for facet in facets(mesh) if abs(facet.normal().y()) < 1.0e-10 and facet.midpoint().y() < 0.0]))])

# geometric values that account for the presence of the aqueduct (useful for interpolating bc between interfaces)
m1 = -aqueductY.max()
m2 = -aqueductY.min()

# membrane permeability
beta_pia = Constant(beta_pia_val/Kscale) # units needed are g/(cm^2*s)
beta_ventricle = Constant(beta_ventricle_val/Kscale)
# smooth step function for interpolating the permeability: on the aqueduct the permeability is assumed to be the same as in the pia mater
beta_interp = interpolate(Expression("1.0 - tanh(10.0*(1 - (x[1] + m2)/(m2 - m1)))", m1 = m1, m2 = m2, degree = 4), V)
beta_aqueduct = (beta_ventricle - beta_pia)*beta_interp + beta_pia

# Initial condition for (u0, p0)
w0 = Function(W)
(u0, p0) = split(w0)

# Variational formulation
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# external pressure at the interfaces
# here pin is the difference between p_ventr and p_sas
pin1 = Expression("sin(2*pi*t)", t = float(dt), degree = 1)
pin2 = Constant(0.0)
interp = interpolate(Expression("(x[1] + m2)/(m2 - m1)", m1 = m1, m2 = m2, degree = 2), V)
pin3 = pin1*interp + (Constant(1.0) - interp)*pin2

# get subdofs
is0 = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
is1 = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())

# remove 2D rigid body motions
def build_nullspace(W, x):
    """Function to build null space for 2D elasticity"""
    rbms = [Constant((0,1)),
	    Constant((1,0)),
	    Expression(('-x[1]','x[0]'), degree = 1)]

    RBMS = [Function(W) for i in xrange(3)]
    rbms = [interpolate(rbm,U) for rbm in rbms]
    for i in xrange(3):
	assign(RBMS[i].sub(0),rbms[i])

    # basis is a dolfin basis for the full system, nullsp is a PETSc null space basis, but only for the displacement components
    # reduced basis is the dolfin version of nullsp
    basis = VectorSpaceBasis([rbm.vector() for rbm in RBMS])
    rbms = [extract_sub_vector(rbm.vector(), 0) for rbm in RBMS]
    reduced_basis = VectorSpaceBasis([PETScVector(item) for item in rbms])
    basis.orthonormalize()
    reduced_basis.orthonormalize()
    
    nullsp = PETSc.NullSpace().create(vectors=[as_backend_type(reduced_basis._sub(i)).vec() for i in xrange(3)])

    return basis, nullsp #, reduced_basis

c0 = Constant(1.0e0) 
# Two-field variational formulation
F = Constant(2.0*mu/lmbda*xscale)*inner(sym(grad(u)), sym(grad(v)))*dx + Constant(xscale)*div(u)*div(v)*dx - p*div(v)*dx \
    + inner(pin1*n,v)*ds(1) \
    + inner(pin2*n,v)*ds(2) \
    + inner(pin3*n,v)*ds(3) \
    - inner(f, v)*dx \
    - div(u - u0)*q*dx \
    - c0*(p - p0)*q*dx \
    - dt*(inner(K*grad(p), grad(q))*dx \
    + beta_ventricle * (p - pin1)*q*ds(1) \
    + beta_pia       * (p - pin2)*q*ds(2) \
    + beta_aqueduct  * (p - pin3)*q*ds(3) \
    - g*q*dx) 

# define solution function
w = Function(W)

uu = Function(U)

# create null space basis
null_space, nullsp = build_nullspace(W, w.vector())

# create PETSc solver
solver = PETSc.KSP().create()
solver.setOptionsPrefix("w_")
solver.setFromOptions()
if fieldsplit_on is True:
    isets = [PETSc.IS().createGeneral(W.sub(i).dofmap().dofs()) for i in xrange(2)]
    solver.pc.setFieldSplitIS(*zip(["0", "1"], isets))

(A, b) = assemble_system(*system(F))
A = as_backend_type(A).mat()
solver.setOperators(A)
solver.setUp()

if fieldsplit_on is True:
    solver.pc.getFieldSplitSubKSP()[0].getOperators()[0].setNullSpace(nullsp)
    solver.pc.getFieldSplitSubKSP()[0].getOperators()[0].setNearNullSpace(nullsp)
    # a bit of a confusing one liner that gets the position of the dofs of the displacement components
    subdofs = [searchsorted(W.sub(0).dofmap().dofs(),W.sub(0).sub(i).dofmap().dofs()).astype("int32") for i in xrange(2)]
    isets0 = [PETSc.IS().createGeneral(subdofs[i]) for i in xrange(2)]
    solver.pc.getFieldSplitSubKSP()[0].pc.setFieldSplitIS(*zip(["0", "1"], isets0))

files = [File("u.pvd"), File("p.pvd")]

for i in xrange(int(10./float(dt)) + 1):
    
    t = (i+1)*float(dt)
    pin1.t = t

    # project the rhs vector into the column space of the linear system matrix so that the
    # linear system admits a solution.
    b = as_backend_type(assemble(rhs(F)))
    null_space.orthogonalize(b)
    b = b.vec()

    if i == 0 or fieldsplit_on is False:
        x = b.duplicate()
    else:
        solver.setInitialGuessNonzero(True)

    solver.solve(b,x)
    w.vector()[:] = x.getArray()

    (u1, p1) = w.split(deepcopy=True)

    p1.vector()[:] = p1.vector().array()*p_scale/10. # dimensionalise and rescale to Pascals
    u1.vector()[:] = u1.vector().array()*u_scale # dimensionalise

    # NOTE this is messy, keep track of some computed values
    print "simulation time: %f" % t
    print "volume change: %f" % (assemble(inner(u1,n)*ds)) # volume change
    print "normal displacement, (ventricles, pia): (%f, %f)" % (assemble(inner(u1,n)*ds(1)), assemble(inner(u1,n)*ds(2))) # normal displacement
    print "normal flux, (ventricles, pia): (%f, %f)" % (assemble(inner(K*Kscale*grad(p1),n)*ds(1)), assemble(inner(K*Kscale*grad(p1),n)*ds(2))) # flux
    print "pressure gradient norm: %f" % (assemble(inner(grad(p1), grad(p1))*dx)**.5) # pressure gradient norm
    print "normalised displacement norm: %f" %  (assemble(inner(u1,u1)*dx)**.5/vol**.5) # normalised displacement norm
    print "normalised pressure norm: %f" % (assemble(p1*p1*dx)**.5/vol**.5) # normalised pressure norm
    #print ((assemble((p1 - pin1)**2.*ds(1))/assemble(Constant(1.0)*ds(1, domain=mesh)))**.5, assemble((p1-pin2)**2.*ds(2))**.5, assemble((p1-pin3)**2.*ds(3))**.5) # normalised pressure jump norm
    print "max absolute displacement: %f" % (abs(u1.vector().array()).max()) # max displacement
    print "max absolute pressure: %f" % (abs(p1.vector().array()).max()) # max pressure

    w0.assign(w)

    files[0] << u1 # output u is in cm
    files[1] << p1 # output p is in Pa

    #import sys; sys.exit(0)
