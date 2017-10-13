# Matteo Croci, October 12 2017

from dolfin import *
from petsc4py import PETSc
from numpy import hstack, unique, array, searchsorted, concatenate, empty

# Setting C++ optimisation flags
parameters['form_compiler']['optimize']=True
parameters['form_compiler']['cpp_optimize']=True
parameters['form_compiler']['representation']='uflacs'
parameters['form_compiler']['cpp_optimize_flags']='-O3 -ffast-math -march=native'

###################################################################################################################
### PETSc options
###################################################################################################################

dolfin.SubSystemsManager.init_petsc()

# outer krylov solver : flexible GMRES
# outer preconditioner: fieldsplit block Gauss-Seidel (multiplicative) for displacement, total pressure and pressure.
# displacement solver : either use cholesky or split the components (multiplicative fieldsplit again)
#                       and use 1 hypre algebraic multigrid (AMG) V-cycle as preconditioner for each components. Best if applied 5 times
# total press solver  : apply preconditioner from paper from Lee, Mardal, Winther
# pressure solver     : apply multiple AMG cycles (5 works well)

opts   = {"w_ksp_type" : "fgmres",
          "w_ksp_atol" : 3.0e-8,
          "w_ksp_rtol" : 1.0e-11,
          "w_ksp_max_it" : 200,
          "w_pc_factor_mat_solver_package" : "mumps",
          "w_ksp_converged_reason" : None,
          "w_ksp_monitor_true_residual" : None,

          "w_pc_type" : "fieldsplit",
          "w_pc_fieldsplit_type" : "multiplicative",
          "w_pc_fieldsplit_0_fields" : 0,
          "w_pc_fieldsplit_1_fields" : 1,
          "w_pc_fieldsplit_2_fields" : 2,

          "w_fieldsplit_0_ksp_type" : "preonly",
          "w_fieldsplit_0_pc_type" : "cholesky",
          "w_fieldsplit_0_pc_factor_mat_solver_package" : "mumps",
          "w_fieldsplit_0_mat_mumps_icntl_24" : 1,

          ## NOTE: comment the three lines above and uncomment those below to change from
          ##       cholesky factorisation (direct solver) to multiplicative fieldsplit preconditioning
          ##       you also need to uncomment lines 366-390.
          #"w_fieldsplit_0_ksp_type" : "richardson",
          #"w_fieldsplit_0_pc_type" : "fieldsplit",
          #"w_fieldsplit_0_ksp_atol" : 1.0e-10,
          #"w_fieldsplit_0_ksp_rtol" : 1.0e-15,
          #"w_fieldsplit_0_ksp_monitor_true_residualx" : None,
          #"w_fieldsplit_0_ksp_converged_reasonx" : None,
          #"w_fieldsplit_0_ksp_max_it" : 5,

          #"w_fieldsplit_0_pc_fieldsplit_type" : "multiplicative",
          #"w_fieldsplit_0_pc_fieldsplit_0_fields" : 0,
          #"w_fieldsplit_0_pc_fieldsplit_1_fields" : 1,
          #"w_fieldsplit_0_pc_fieldsplit_2_fields" : 2,

          #"w_fieldsplit_0_fieldsplit_ksp_type" : "preonly",
          #"w_fieldsplit_0_fieldsplit_ksp_atol" : 1.0e-10,
          #"w_fieldsplit_0_fieldsplit_ksp_rtol" : 1.0e-15,
          #"w_fieldsplit_0_fieldsplit_ksp_monitor_true_residualx" : None,
          #"w_fieldsplit_0_fieldsplit_ksp_converged_reasonx" : None,
          #"w_fieldsplit_0_fieldsplit_pc_factor_mat_solver_package" : "mumps",
          #"w_fieldsplit_0_fieldsplit_ksp_max_it" : 1,
          #"w_fieldsplit_0_fieldsplit_pc_type" : "hypre",
          #"w_fieldsplit_0_fieldsplit_pc_hypre_type" : "boomeramg",

          "w_fieldsplit_1_ksp_type" : "preonly",
          "w_fieldsplit_1_ksp_atol" : 1.0e-10,
          "w_fieldsplit_1_ksp_rtol" : 1.0e-15,
          "w_fieldsplit_1_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_1_ksp_converged_reasonx" : None,
          "w_fieldsplit_1_pc_factor_mat_solver_package" : "mumps",
          "w_fieldsplit_1_ksp_max_it" : 1,
          "w_fieldsplit_1_pc_type" : "mat",
          "w_fieldsplit_1_pc_hypre_type" : "boomeramg",

          "w_fieldsplit_2_ksp_type" : "richardson",
          "w_fieldsplit_2_ksp_atol" : 1.0e-10,
          "w_fieldsplit_2_ksp_rtol" : 1.0e-15,
          "w_fieldsplit_2_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_2_ksp_converged_reasonx" : None,
          "w_fieldsplit_2_pc_factor_mat_solver_package" : "mumps",
          "w_fieldsplit_2_ksp_max_it" : 5,
          "w_fieldsplit_2_pc_type" : "hypre",
          "w_fieldsplit_2_pc_hypre_type" : "boomeramg",
          }

# set PETSc options
petsc_opts = PETSc.Options()
for key in opts:
    petsc_opts[key] = opts[key]

###################################################################################################################
### Auxiliary functions
###################################################################################################################

def extract_sub_matrix(A, subspace_in, subspace_out):
    Amat  = as_backend_type(A).mat()

    subis_in   = PETSc.IS()
    subdofs_in = W.sub(subspace_in).dofmap().dofs()
    subis_in.createGeneral(subdofs_in)

    subis_out   = PETSc.IS()
    subdofs_out = W.sub(subspace_out).dofmap().dofs()
    subis_out.createGeneral(subdofs_out)

    # NOTE: the following line might not work, depending on the petsc4py version
    #       if so, replace with createSubMatrix
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

###################################################################################################################
### Load mesh and markers
###################################################################################################################

## NOTE: uncomment the next block to use the collins27 mesh. You might have to modify the path.
#path_to_mesh_file = "../../../brain-mesh-2015/collins27/"
#mesh = Mesh(mpi_comm_world(), path_to_mesh_file + "whitegray.xml.gz")
#hdf = HDF5File(mesh.mpi_comm(), "collins_markers.h5", "r")
#markers = MeshFunction("size_t", mesh)
#hdf.read(markers, "/markers")
#boundaries = FacetFunction("size_t", mesh)
#hdf.read(boundaries, "/boundaries")
## marker_number is the marker number assigned to the grey matters dofs, for the collins27 this is 3,
## for the spherical brain mesh this is 1.
#marker_number = 3

# NOTE: the following loads the spherical brain mesh and relative whitegrey matter markers generated in gmesh
hdf = HDF5File(mpi_comm_world(), "mesh_and_markers.h5", "r")
mesh = Mesh(mpi_comm_world())
hdf.read(mesh, "/mesh", False)
markers = MeshFunction("size_t", mesh) 
hdf.read(markers, "/markers")
boundaries = FacetFunction("size_t", mesh)
hdf.read(boundaries, "/boundaries")
# marker_number is the marker number assigned to the grey matters dofs, for the collins27 this is 3,
# for the spherical brain mesh this is 1.
marker_number = 1

ds = Measure("ds", subdomain_data = boundaries)

###################################################################################################################
### Setup parameters
###################################################################################################################

# domain volume, needed later for output functionals and preconditioner
vol = assemble(Constant(1.0)*dx(domain = mesh))

# Define other material parameters. The length unit is cm.
E = 3156. # conversion_factor(from Pa to g/mm/s^2) = 1
nu = 0.4983 
mu = E/(2.*(1+nu)) # conversion from E
lmbda = 2.*mu*nu/(1.-2.*nu) # lame' parameters
# Define permeabilities and divide them by the ISF viscosity (similar as for water). 
# this is m^2*[conversion factor to mm^2]/([ISF viscosity in g/(mm*s)]/[conversion factor from mm to cm])
Kgrey_val  = 3.0e-12*1.0e6/(0.658e-3) # permeabilities. There is a factor of 20 that is added to make results look nicer :)
Kwhite_val = 1.4e-12*1.0e6/(0.658e-3)
pout_val = 0.0*133.32
pin_val  = 0.0*133.32 + 0.1*133.32 # CSF pressure
# NOTE: the permeability values are unknown
beta_pia_val = 1e-3 # membrane permeability at the pia interface, units needed are g/(mm^2*s)
beta_ventricle_val = 1e-5 # membrane permeability at the ventricular interface
alpha = 0.98
n = 0.8 # porosity
Ks = E/(3.*(1.-2.*nu))
s0_val = (1.-n)/Ks 

###################################################################################################################
### Setup parameter scalings
###################################################################################################################

# scale factors for nondimensionalisation
xscale = 1.0
Kscale = xscale**2/(2.*mu)
p_scale = pin_val
u_scale = pin_val/(2.*mu)*xscale
s_scale = alpha**2./(2.*mu)
lmbdatilde = lmbda/(2.*mu)
invlmbda = Constant(1.0/lmbdatilde)

print "nondimentionalised input parameters, (lmbda/(2*mu)*xscale, Kwhite, beta, s0): (%f, %f, %f, %f)" % (lmbda/(2.*mu)*xscale, Kgrey_val/Kscale, beta_pia_val/Kscale, s0_val/s_scale)

###################################################################################################################
### Setup characteristic function for distinguishing grey and white matter
###################################################################################################################

# characteristic function of the grey matter regions
DG = FunctionSpace(mesh,'DG',0)
dofmap = DG.dofmap()
grey_matter_dofs = concatenate([dofmap.cell_dofs(cell.index()) for cell in cells(mesh) if markers[cell] == marker_number])
K = interpolate(Constant(Kwhite_val/Kscale),DG)
K_vec = K.vector()
values = K_vec.get_local()
values[grey_matter_dofs] = Kgrey_val/Kscale
K_vec.set_local(values)

###################################################################################################################
### Setup zero forcing terms, time step and FacetNormal
###################################################################################################################

# time step
dt = Constant(0.01)
# Volume force/source
f = Constant((0.0, 0.0, 0.0))
g = Constant(0.0)

normal = FacetNormal(mesh)

###################################################################################################################
### Setup function spaces
###################################################################################################################

# Define finite element spaces
Velem = VectorElement("CG", mesh.ufl_cell(), 2)
Qelem = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = MixedElement([Velem, Qelem, Qelem])
W = FunctionSpace(mesh, ME)
U = FunctionSpace(mesh, Velem)
V = FunctionSpace(mesh, Qelem)

# Initial condition for (u0, P0, p0)
w0 = Function(W)
(u0, P0, p0) = split(w0)

# Variational formulation, u - displacement, P - total pressure, p - pressure
(u, P, p) = TrialFunctions(W)
(v, Q, q) = TestFunctions(W)

###################################################################################################################
### Setup parameters and functions for Boundary Conditions
###################################################################################################################

# membrane permeability
beta_pia = Constant(beta_pia_val/Kscale) # units needed are g/(cm^2*s)
beta_ventricle = Constant(beta_ventricle_val/Kscale)
# external pressure at the interfaces NOTE: 1 here is pia and 2 here is ventricles!!! 
pin1 = Expression("pin_val/p_scale*sin(2*pi*t)",  t = float(dt), p_scale = p_scale, pin_val = pin_val, degree = 1)
pin2 = Expression("pout_val/p_scale*sin(2*pi*t)", t = float(dt), p_scale = p_scale, pout_val = pout_val, degree = 1)
#pin1 = Constant(pin_val/p_scale) # pia
#pin2 = Constant(pout_val/p_scale) # ventricles

###################################################################################################################
### Create null space for rigid body motions
###################################################################################################################

# remove 3D rigid body motions
def build_nullspace(W, x):
    """Function to build null space for 2D elasticity"""
    rbms = [Constant((0,0,1)),
	    Constant((0,1,0)),
	    Constant((1,0,0)),
	    Expression(('-x[1]','x[0]','0.0'), degree=1),
	    Expression(('x[2]','0.0','x[0]'),  degree=1),
	    Expression(('0.0','-x[2]','x[1]'), degree=1)]

    RBMS = [Function(W) for i in xrange(6)]
    rbms = [interpolate(rbm,U) for rbm in rbms]
    for i in xrange(6):
	assign(RBMS[i].sub(0),rbms[i])

    # basis is a dolfin basis for the full system, nullsp is a PETSc null space basis, but only for the displacement components
    # reduced basis is the dolfin version of nullsp
    basis = VectorSpaceBasis([rbm.vector() for rbm in RBMS])
    rbms = [extract_sub_vector(rbm.vector(), 0) for rbm in RBMS]
    reduced_basis = VectorSpaceBasis([PETScVector(item) for item in rbms])
    basis.orthonormalize()
    reduced_basis.orthonormalize()
    
    nullsp = PETSc.NullSpace().create(vectors=[as_backend_type(reduced_basis._sub(i)).vec() for i in xrange(6)])

    return basis, nullsp #, reduced_basis


# define solution function
w = Function(W)

uu = Function(U)

# create null space basis
null_space, nullsp = build_nullspace(W, w.vector())

###################################################################################################################
### Three field weak formulation
###################################################################################################################

# Three-field variational formulation (Lee, Mardal, Winther)
displacement = inner(sym(grad(u)), sym(grad(v)))*dx - P*div(v)*dx \
             + inner(pin1*normal,v)*ds(1) \
             + inner(pin2*normal,v)*ds(2) \
             - inner(f, v)*dx

total_pressure = - div(u)*Q*dx - invlmbda*P*Q*dx + Constant(alpha)*invlmbda*p*Q*dx - g*Q*dx

pressure = Constant(alpha)*invlmbda*(P - P0)*q*dx \
         - Constant(alpha**2)*(Constant(s0_val/s_scale) + invlmbda)*(p - p0)*q*dx \
         - dt*(inner(K*grad(p), grad(q))*dx \
         + beta_pia       * (p - pin1)*q*ds(1) \
         + beta_ventricle * (p - pin2)*q*ds(2) \
         - g*q*dx) 

F = displacement + total_pressure + pressure

###################################################################################################################
### Create preconditioner for the total pressure as in Lee, Mardal, Winther
###################################################################################################################

mm  = extract_sub_vector(assemble(Q*dx), 1)
dd  = extract_sub_matrix(assemble(P*Q*dx), 1, 1)
szs = dd.getSizes()
dd  = dd.getDiagonal()
dd.reciprocal()
class TotalPressurePreconditioner(object):
    def mult(dummyself, mat, x, y):
        mm.axpy(x.sum()*(lmbdatilde - 1.0)/vol, x)
        y.pointwiseMult(x, dd)

Ptotal = PETSc.Mat().createPython(szs, TotalPressurePreconditioner())

###################################################################################################################
### Assemble system and set up the Krylov solver, i.e. the PETSc KSP
###################################################################################################################

# create PETSc solver
solver = PETSc.KSP().create()
solver.setOptionsPrefix("w_")
solver.setFromOptions()
# set fieldsplit fields
isets = [PETSc.IS().createGeneral(W.sub(i).dofmap().dofs()) for i in xrange(3)]
solver.pc.setFieldSplitIS(*zip(["0", "1", "2"], isets))

# asseble the system and give the assembled system to the KSP
(A, b) = assemble_system(*system(F))
A = as_backend_type(A).mat()
solver.setOperators(A)
solver.setUp()

# set null space and near-null space
solver.pc.getFieldSplitSubKSP()[0].getOperators()[0].setNullSpace(nullsp)
solver.pc.getFieldSplitSubKSP()[0].getOperators()[0].setNearNullSpace(nullsp)
## the block size of the displacement field is lost by dolfin, so we reset it
## NOTE: the following line might not work, depending on the petsc4py version
#solver.pc.getFieldSplitSubKSP()[0].getOperators()[0].setBlockSize(3)

# assign the total pressure preconditioner to the KSP
AA = solver.pc.getFieldSplitSubKSP()[1].getOperators()[0]
solver.pc.getFieldSplitSubKSP()[1].setOperators(AA, Ptotal)

##NOTE: uncomment the following block if using an inner fieldsplit preconditioner for the displacement block
#################################################################################################################
## WARNING: the following is confusing, and it is only needed if we do not use cholesky/LU for the displacement,
##          but we use an additional fieldsplit for the displacement field. Nevermind if this part is not clear.
##
## To set up the fieldsplit for the displacement we need to tell PETSc the position of the dofs of each
## displacement component in the displacement matrix block. To do so, we need to first have the list of all
## displacement dofs. These are split between processors so we need to use mpi to gather them on each worker.
#comm = mpi_comm_world().tompi4py()
#if comm.size == 0:
#    # if we are only using one processor, then we have all the displacement dofs
#    recvbuf = W.sub(0).dofmap().dofs()
#    # the next lines obtain the same result of the above line, but they work in parallel
#else:
#    # compute the dofs owned by each processor, then send them along to all the other ones
#    sendbuf = W.sub(0).dofmap().dofs()
#    sendcounts = array(comm.allgather(len(sendbuf)))
#    recvbuf = empty(sum(sendcounts), dtype='int32')
#    comm.Allgatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts))
#
## find the position of the component dofs in the displacement dofs and set the relative fieldsplit fields up
#subdofs = [searchsorted(recvbuf,W.sub(0).sub(i).dofmap().dofs()).astype("int32") for i in xrange(3)]
#isets0 = [PETSc.IS().createGeneral(subdofs[i]) for i in xrange(3)]
#solver.pc.getFieldSplitSubKSP()[0].pc.setFieldSplitIS(*zip(["0", "1", "2"], isets0))
#################################################################################################################

###################################################################################################################
### Main loop
###################################################################################################################

files = [File("u.pvd"), File("p.pvd")]

for i in xrange(int(10./float(dt)) + 1):
    
    t = (i+1)*float(dt)
    pin1.t = t
    pin2.t = t

    # project the rhs vector into the column space of the linear system matrix so that the
    # linear system admits a solution.
    b = as_backend_type(assemble(rhs(F)))
    null_space.orthogonalize(b)
    b = b.vec()

    # reuse the previous iteration as initial guess for the next step
    if i == 0:
        x = b.duplicate()
    else:
        solver.setInitialGuessNonzero(True)

    solver.solve(b,x)
    print "SOLVED!!!!"
    w.vector()[:] = x.getArray()

    (u1, P1, p1) = w.split(deepcopy=True)

    p1.vector().set_local(p1.vector().array()*p_scale) # dimensionalise and rescale to Pascals
    u1.vector().set_local(u1.vector().array()*u_scale) # dimensionalise

    # NOTE this is messy, keep track of some computed values
    print "simulation time: %f" % t
    print "volume change, pia: %f, ventricles: %f, total: %f" % (assemble(inner(u1,normal)*ds(1)), assemble(inner(u1,normal)*ds(2)), assemble(inner(u1,normal)*ds)) # volume change
    print "average normal displacement, (pia, ventricles): (%f, %f)" % (assemble(inner(u1,normal)*ds(1))/(2.*DOLFIN_PI*30.), assemble(inner(u1,normal)*ds(2))/(2.*DOLFIN_PI*100.)) # normal displacement
    print "average normal flux, (pia, ventricles, total unaveraged): (%f, %f, %f)" % (assemble(inner(K*Kscale*grad(p1),normal)*ds(1))/(2.*DOLFIN_PI*30.), assemble(inner(K*Kscale*grad(p1),normal)*ds(2))/(2.*DOLFIN_PI*100.), assemble(inner(K*Kscale*grad(p1),normal)*ds)) # flux
    print "pressure gradient norm: %f" % (assemble(inner(grad(p1), grad(p1))*dx)**.5) # pressure gradient norm
    print "normalised displacement norm: %f" %  (assemble(inner(u1,u1)*dx)**.5/vol**.5) # normalised displacement norm
    print "normalised pressure norm: %f" % (assemble(p1*p1*dx)**.5/vol**.5) # normalised pressure norm
    print "max absolute displacement: %f" % (abs(u1.vector().array()).max()) # max displacement
    print "max absolute pressure: %f" % (abs(p1.vector().array()).max()) # max pressure

    files[0] << u1 # output u is in mm
    files[1] << p1 # output p is in Pa

    w0.assign(w)

    #import sys; sys.exit(0)
