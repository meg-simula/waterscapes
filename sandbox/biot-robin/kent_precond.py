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
          "w_ksp_max_it" : 2000,
          "w_pc_factor_mat_solver_package" : "mumps",
          "w_ksp_converged_reason" : None,
          "w_ksp_monitor_true_residual" : None,

          "w_pc_type" : "fieldsplit",
          "w_pc_fieldsplit_type" : "additive",
          "w_pc_fieldsplit_0_fields" : 0,
          "w_pc_fieldsplit_1_fields" : 1,
          "w_pc_fieldsplit_2_fields" : 2,
          "w_pc_fieldsplit_3_fields" : 3,

          "w_fieldsplit_0_ksp_type" : "preonly",
          "w_fieldsplit_0_pc_type" : "gamg",
          #"w_fieldsplit_0_pc_gamg_agg_nsmooths" : 3,
          #"w_fieldsplit_0_pc_gamg_coarse_eq_limit" : 50000,
          "w_fieldsplit_0_ksp_atol" : 1.0e-10,
          "w_fieldsplit_0_ksp_rtol" : 1.0e-15,
          "w_fieldsplit_0_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_0_ksp_converged_reasonx" : None,
          "w_fieldsplit_0_ksp_max_it" : 5,
          "w_fieldsplit_0_pc_gamg_use_parallel_coarse_grid_solver": True,

          "w_fieldsplit_1_ksp_type" : "cg",
          "w_fieldsplit_1_ksp_atol" : 1.0e-10,
          "w_fieldsplit_1_ksp_rtol" : 1.0e-15,
          "w_fieldsplit_1_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_1_ksp_converged_reasonx" : None,
          "w_fieldsplit_1_pc_factor_mat_solver_package" : "mumps",
          "w_fieldsplit_1_ksp_max_it" : 10,
          "w_fieldsplit_1_pc_type" : "jacobi",
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

          "w_fieldsplit_3_ksp_type" : "preonly",
          "w_fieldsplit_3_ksp_max_it" : 1,
          "w_fieldsplit_3_ksp_monitor_true_residualx" : None,
          "w_fieldsplit_3_ksp_converged_reasonx" : None,
          "w_fieldsplit_3_pc_type" : "jacobi",
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

def rm_basis(mesh):
    import numpy as np
    '''6 functions(Expressions) that are the rigid motions of the body.'''
    x = SpatialCoordinate(mesh)
    dX = dx(domain=mesh)
    # Center of mass
    c = np.array([assemble(xi*dX) for xi in x])
    volume = assemble(Constant(1)*dX)
    c /= volume
    c_ = c
    c = Constant(c)

    # Gram matrix of rotations around canonical axis and center of mass
    R = np.zeros((3, 3))

    ei_vectors = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0, 1))]
    for i, ei in enumerate(ei_vectors):
        R[i, i] = assemble(inner(cross(x-c, ei), cross(x-c, ei))*dX)
        for j, ej in enumerate(ei_vectors[i+1:], i+1):
            R[i, j] = assemble(inner(cross(x-c, ei), cross(x-c, ej))*dX)
            R[j, i] = R[i, j]

    # Eigenpairs
    eigw, eigv = np.linalg.eigh(R)      
    if np.min(eigw) < 1E-8: warning('Small eigenvalues %g.' % np.min(eigw))
    eigv = eigv.T

    # Translations: ON basis of translation in direction of rot. axis
    translations = [Constant(v/sqrt(volume)) for v in eigv]

    # Rotations using the eigenpairs
    # C0, C1, C2 = c.values()
    C0, C1, C2 = c_

    def rot_axis_v(pair):
        '''cross((x-c), v)/sqrt(w) as an expression'''
        v, w = pair
        return Expression(('((x[1]-C1)*v2-(x[2]-C2)*v1)/A',
                           '((x[2]-C2)*v0-(x[0]-C0)*v2)/A',
                           '((x[0]-C0)*v1-(x[1]-C1)*v0)/A'),
                           C0=C0, C1=C1, C2=C2, 
                           v0=v[0], v1=v[1], v2=v[2], A=sqrt(w),
                           degree=1)

    # Roations are described as rot around v-axis centered in center of gravity 
    rotations = map(rot_axis_v, zip(eigv, eigw))

    Z = translations + rotations
    return Z

def rescale_mesh(mesh):
    X = mesh.coordinates()
    x = SpatialCoordinate(mesh)
    dX = dx(domain=mesh)
    # Center of mass
    c = array([assemble(xi*dX) for xi in x])
    volume = assemble(Constant(1)*dX)
    xscale = pow(volume, 1./mesh.geometry().dim())
    rescaled_X = (X - c/volume)/xscale
    rescaled_mesh = Mesh(mesh)
    rescaled_mesh.coordinates()[:] = rescaled_X
    rescaled_mesh.bounding_box_tree().build(rescaled_mesh)
    return rescaled_mesh, xscale

###################################################################################################################
### Load mesh and markers
###################################################################################################################

## NOTE: uncomment the next block to use the collins27 mesh. You might have to modify the path.
#path_to_mesh_file = "../../../brain-mesh-2015/collins27/"
#old_mesh = Mesh(mpi_comm_world(), path_to_mesh_file + "whitegray.xml.gz")
#
#mesh, xscale = rescale_mesh(old_mesh)
#
#hdf = HDF5File(mpi_comm_world(), "collins_markers.h5", "r")
#markers = MeshFunction("size_t", mesh)
#hdf.read(markers, "/markers")
#boundaries = FacetFunction("size_t", mesh)
#old_boundaries = FacetFunction("size_t", old_mesh)
#hdf.read(boundaries, "/boundaries")
#hdf.read(old_boundaries, "/boundaries")
## marker_number is the marker number assigned to the grey matters dofs, for the collins27 this is 3,
## for the spherical brain mesh this is 1.
#marker_number = 3

# NOTE: the following loads the spherical brain mesh and relative whitegrey matter markers generated in gmesh
hdf = HDF5File(mpi_comm_world(), "mesh_and_markers.h5", "r")
old_mesh = Mesh(mpi_comm_world())
hdf.read(old_mesh, "/mesh", False)

mesh, xscale = rescale_mesh(old_mesh)

markers = MeshFunction("size_t", mesh) #, path_to_mesh_file + "whitegray_markers.xml.gz")
hdf.read(markers, "/markers")
boundaries = FacetFunction("size_t", mesh)
old_boundaries = FacetFunction("size_t", old_mesh)
hdf.read(boundaries, "/boundaries")
hdf.read(old_boundaries, "/boundaries")
# marker_number is the marker number assigned to the grey matters dofs, for the collins27 this is 3,
# for the spherical brain mesh this is 1.
marker_number = 1

ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("ds", subdomain_data = old_boundaries)

###################################################################################################################
### Setup parameters
###################################################################################################################

# domain volume, needed later for output functionals and preconditioner
vol = assemble(Constant(1.0)*dx(domain = old_mesh))

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
Kscale = xscale**2/(2.*mu)
beta_scale = xscale/(2.*mu) # this is Kscale/xscale
p_scale = pin_val
u_scale = pin_val/(2.*mu)*xscale
s_scale = alpha**2./(2.*mu)
lmbdatilde = lmbda/(2.*mu)
invlmbda = Constant(1.0/lmbdatilde)

print "nondimentionalised input parameters, (lmbda/(2*mu), Kwhite, beta, s0): (%f, %f, %f, %f)" % (lmbda/(2.*mu), Kgrey_val/Kscale, beta_pia_val/beta_scale, s0_val/s_scale)

###################################################################################################################
### Setup characteristic function for distinguishing grey and white matter
###################################################################################################################

# characteristic function of the grey matter regions
DG = FunctionSpace(mesh,'DG',0)
old_DG = FunctionSpace(old_mesh,'DG',0)
dofmap = DG.dofmap()
grey_matter_dofs = concatenate([dofmap.cell_dofs(cell.index()) for cell in cells(mesh) if markers[cell] == marker_number])
K = interpolate(Constant(Kwhite_val/Kscale),DG)
K_vec = K.vector()
values = K_vec.get_local()
values[grey_matter_dofs] = Kgrey_val/Kscale
K_vec.set_local(values)

old_K = Function(old_DG)
old_K.vector().set_local(K.vector().array())

###################################################################################################################
### Setup zero forcing terms, time step and FacetNormal
###################################################################################################################

# time step
dt = Constant(0.01)
# Volume force/source
f = Constant((0.0, 0.0, 0.0))
g = Constant(0.0)

normal = FacetNormal(mesh)
old_normal = FacetNormal(old_mesh)

###################################################################################################################
### Setup function spaces
###################################################################################################################

# Define finite element spaces
Velem = VectorElement("CG", mesh.ufl_cell(), 2)
Qelem = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Relem = VectorElement('R', mesh.ufl_cell(), 0, dim = 6)
ME = MixedElement([Velem, Qelem, Qelem, Relem])
W = FunctionSpace(mesh, ME)
U = FunctionSpace(mesh, Velem)
V = FunctionSpace(mesh, Qelem)

# old mesh function
Wold = FunctionSpace(old_mesh, ME)
wold = Function(Wold)
(uold, Pold, pold, rold) = wold.split(deepcopy=True)

# Initial condition for (u0, P0, p0, r0)
w0 = Function(W)
(u0, P0, p0, r0) = split(w0)

# Variational formulation, u - displacement, P - total pressure, p - pressure, r - Lagrange multiplier
(u, P, p, r) = TrialFunctions(W)
(v, Q, q, s) = TestFunctions(W)

###################################################################################################################
### Setup parameters and functions for Boundary Conditions
###################################################################################################################

# membrane permeability
beta_pia = Constant(beta_pia_val/beta_scale) # units needed are g/(cm^2*s)
beta_ventricle = Constant(beta_ventricle_val/beta_scale)
# external pressure at the interfaces NOTE: 1 here is pia and 2 here is ventricles!!! 
pin1 = Expression("pin_val/p_scale*sin(2*pi*t)",  t = float(dt), p_scale = p_scale, pin_val = pin_val, degree = 1)
pin2 = Expression("pout_val/p_scale*sin(2*pi*t)", t = float(dt), p_scale = p_scale, pout_val = pout_val, degree = 1)
#pin1 = Constant(pin_val/p_scale) # pia
#pin2 = Constant(pout_val/p_scale) # ventricles

###################################################################################################################
### Create null space for rigid body motions
###################################################################################################################

rm = rm_basis(mesh)

s = sum([s[i]*rm[i] for i in range(len(rm))])
r = sum([r[i]*rm[i] for i in range(len(rm))])

# define solution function
w = Function(W)

uu = Function(U)

###################################################################################################################
### Three field weak formulation
###################################################################################################################

# Three-field variational formulation (Lee, Mardal, Winther)
displacement = inner(sym(grad(u)), sym(grad(v)))*dx - P*div(v)*dx \
             + inner(pin1*normal,v)*ds(1) \
             + inner(pin2*normal,v)*ds(2) \
             - inner(f, v)*dx

lagrange_mult = inner(u,s)*dx + inner(v,r)*dx

total_pressure = - div(u)*Q*dx - invlmbda*P*Q*dx + Constant(alpha)*invlmbda*p*Q*dx - g*Q*dx

pressure = Constant(alpha)*invlmbda*(P - P0)*q*dx \
         - Constant(alpha**2)*(Constant(s0_val/s_scale) + invlmbda)*(p - p0)*q*dx \
         - dt*(inner(K*grad(p), grad(q))*dx \
         + beta_pia       * (p - pin1)*q*ds(1) \
         + beta_ventricle * (p - pin2)*q*ds(2) \
         - g*q*dx) 

F = displacement + lagrange_mult + total_pressure + pressure

###################################################################################################################
### Assemble system and set up the Krylov solver, i.e. the PETSc KSP
###################################################################################################################

Pform = inner(sym(grad(u)), sym(grad(v)))*dx + Constant(1.0)*inner(u,v)*dx - inner(f,v)*dx + inner(r,s)*dx + P*Q*dx -g*Q*dx - inner(f,s)*dx

Ppress_form = Constant(alpha**2)*(Constant(s0_val/s_scale) + invlmbda)*p*q*dx + dt*inner(K*grad(p), grad(q))*dx -dt*g*q*dx

Precond = as_backend_type(assemble_system(*system(Pform + Ppress_form))[0]).mat()

###################################################################################################################
### Create null space for rigid body motions
###################################################################################################################

# remove 3D rigid body motions
def build_nullspace(W, x):
    """Function to build null space for 2D elasticity"""

    rbms = rm_basis(W.mesh())

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

# create null space basis
null_space, nullsp = build_nullspace(W, w.vector())

###################################################################################################################
### Assemble system and set up the Krylov solver, i.e. the PETSc KSP
###################################################################################################################

# create PETSc solver
solver = PETSc.KSP().create()
solver.setOptionsPrefix("w_")
solver.setFromOptions()

# set fieldsplit fields
isets = [PETSc.IS().createGeneral(W.sub(i).dofmap().dofs()) for i in xrange(4)]
solver.pc.setFieldSplitIS(*zip(["0", "1", "2", "3"], isets))

# asseble the system and give the assembled system to the KSP
(A, b) = assemble_system(*system(F))
A = as_backend_type(A).mat()
#solver.setOperators(A)
solver.setOperators(A,Precond)
solver.setUp()

solver.pc.getFieldSplitSubKSP()[0].getOperators()[0].setNearNullSpace(nullsp)

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
    b = as_backend_type(assemble(rhs(F))).vec()

    # reuse the previous iteration as initial guess for the next step
    if i == 0:
        x = b.duplicate()
    else:
        solver.setInitialGuessNonzero(True)

    solver.solve(b,x)
    print "SOLVED!!!!"
    w.vector()[:] = x.getArray()

    (u1, P1, p1, r1) = w.split(deepcopy=True)

    pold.vector().set_local(p1.vector().array()*p_scale) # dimensionalise and rescale to Pascals
    uold.vector().set_local(u1.vector().array()*u_scale) # dimensionalise

    # NOTE this is messy, keep track of some computed values
    print "simulation time: %f" % t
    print "volume change, pia: %f, ventricles: %f, total: %f" % (assemble(inner(uold,old_normal)*dS(1)), assemble(inner(uold,old_normal)*dS(2)), assemble(inner(uold,old_normal)*dS)) # volume change
    print "average normal displacement, (pia, ventricles): (%f, %f)" % (assemble(inner(uold,old_normal)*dS(1))/(2.*DOLFIN_PI*30.), assemble(inner(uold,old_normal)*dS(2))/(2.*DOLFIN_PI*100.)) # normal displacement
    print "average normal flux, (pia, ventricles, total unaveraged): (%f, %f, %f)" % (assemble(inner(old_K*Kscale*grad(pold),old_normal)*dS(1))/(2.*DOLFIN_PI*30.), assemble(inner(old_K*Kscale*grad(pold),old_normal)*dS(2))/(2.*DOLFIN_PI*100.), assemble(inner(old_K*Kscale*grad(pold),old_normal)*dS)) # flux
    print "pressure gradient norm: %f" % (assemble(inner(grad(pold), grad(pold))*dx)**.5) # pressure gradient norm
    print "normalised displacement norm: %f" %  (assemble(inner(uold,uold)*dx)**.5/vol**.5) # normalised displacement norm
    print "normalised pressure norm: %f" % (assemble(pold*pold*dx)**.5/vol**.5) # normalised pressure norm
    print "max absolute displacement: %f" % (abs(uold.vector().array()).max()) # max displacement
    print "max absolute pressure: %f" % (abs(pold.vector().array()).max()) # max pressure

    files[0] << uold # output u is in mm
    files[1] << pold # output p is in Pa

    w0.assign(w)

    #import sys; sys.exit(0)
