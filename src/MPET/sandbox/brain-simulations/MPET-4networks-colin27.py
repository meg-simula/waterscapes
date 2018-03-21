""" 4 networks MPET experiment:
    1. CSF, entracellular
    2. Arterial
    3. Venous
    4. Capillar
"""

from __future__ import print_function

__author__ = "Eleonora Piersanti (eleonora@simula.no), 2016-2017"
__all__ = []

# Modified by Marie E. Rognes, 2018

from mpet import *
#from mshr import *

import time as timemodule

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# FIXME
# Code for defining prescribed boundary CSF pressure
cpp_code = """
[&]() {
    if (std::abs(x[0]) <= x_mid && std::abs(x[1]) <= x_mid && std::abs(x[2]) <= x_mid) {
        return mmHg2Pa*(5.0 + 2.0*sin(2.0*pi*time));
    } else {
        return mmHg2Pa*(5.0 + (2.0 + delta)*sin(2.0*pi*time));
    }
}()
"""

def create_mesh():

    # Read mesh
    mesh = Mesh()
    #file = HDF5File(MPI.comm_world, "colin27_whitegray_boundaries.h5", "r")
    file = HDF5File(MPI.comm_world, "colin27_coarse_boundaries.h5", "r")
    # Coarse Colin27 mesh yields 666 198 dofs (about 100 K cells)
    file.read(mesh, "/mesh", False)

    # Read white/gray markers
    D = mesh.topology().dim()
    markers = MeshFunction("size_t", mesh, D) # Not defined for coarse mesh
    #file.read(markers, "/markers")

    # Read skull/ventricle markers
    boundaries = MeshFunction("size_t", mesh, D-1)
    file.read(boundaries, "/boundaries")
    file.close()

    return mesh, markers, boundaries

def mpet_solve(mesh, markers, boundaries,
               M=8, theta=1.0, nu=0.35,
               formulation_type="standard",
               solver_type="direct"):
    "M the number of time steps."
    
    # Define end time T and timestep dt
    time = Constant(0.0)
    T = 1.0
    dt = float(T/M)

    # # Non-scaled parameters from Ventikos
    # # s_24 = 1.5e-19
    # # s_43 = 1.5e-19
    # # s_13 = 1.0e-13 
    # # s_41 = 2.0e-19

    # Pressure ordering:
    # e, a, v, c <-> 1, 2, 3, 4
    # Define material parameters in MPET equations
    A = 4
    (c_e, c_a, c_v, c_c) = (3.9e-4, 2.9e-4, 1.5e-5, 2.9e-4)
    c = (c_e, c_a, c_v, c_c) # In accordance with Vardakis et al, 2017
    print("c = ", c)
    alpha = (0.49, 0.25, 0.01, 0.25) # In accordance with Vardakis et al, 2017
    print("alpha = ", alpha)

    kappa = (1.4e-14, 1.e-10, 1.e-10, 1.e-10) # Vardakis et al, 2016, Oedema
    eta = (8.9e-4, 2.67e-3, 2.67e-3, 2.67e-3)
    scaling = 1.e6 # Scaling from Vardakis values to mm, g, s
    K = [kappa[i]/eta[i]*scaling for i in range(4)]
    print("K = ", ["%0.3g" % K_j for K_j in K])
    
    # Define transfer coefficient and transfer matrix
    s_24 = 1.e-3 # s_a->c and vice versa
    s_43 = 1.e-3 # s_c->v and vice versa
    s_41 = 1.e-3 # s_c->e and vice versa
    s_13 = 1.e-3 # s_e->v and vice versa
    S = ((0.0, 0.0, s_13, s_41),
         (0.0, 0.0, 0.0, s_24),
         (s_13, 0.0, 0.0, s_43),
         (s_41, s_24, s_43, 0.0))

    # Elastic parameters
    E = 1500  # 584 is a bit low, Eleonora looks at literature or
              # reviews with Vegard cf Bitbucket regarding value. Maybe 1500?
    nu = nu
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Setting up MPET problem")

    problem = MPETProblem(mesh, time, params=params)

    # Boundary conditions for the displacement
    problem.u_bar = Constant((0.0, 0.0, 0.0))

    # Boundary condition for the CSF pressure
    mmHg2Pa = 133.32
    delta = 0.012 # mmHg

    # FIXME
    x_mid = 100
    p_CSF = Expression(cpp_code, delta=delta, mmHg2Pa=mmHg2Pa, time=time,
                       x_mid=x_mid, degree=0)
    
    # Boundary condition for the arterial pressure
    p_BPA = Expression("mmHg2Pa*(70.0 + 10.0*sin(2.0*pi*t))",
                       mmHg2Pa=mmHg2Pa, t=time, degree=0)

    # Boundary condition for the venous pressure
    p_BPV = Constant(mmHg2Pa*6.0)

    # Initial condition for the capillary compartment. 
    p_CAP = Constant(mmHg2Pa*(6.0 + 70)/2)
    
    # Collect pressure boundary conditions. Note that we send the
    # p_CAP here, not to use it as a boundary condition, but for
    # convenience in prescribing initial conditions later.
    problem.p_bar = [p_CSF, p_BPA, p_BPV, p_CAP]

    # Mark boundaries for the momentum equation (0 is Dirichlet, 1 is
    # Neumann)
    SKULL = 1
    VENTRICLES = 2
    DIRICHLET = 0
    NEUMANN = 1

    # Iterate over boundary markers, and transfer markers
    info("Transferring boundary markers")
    for i in range(len(boundaries.array())):
        # Boundary conditions on the skull for the different
        # compartments: Dirichlet (0) conditions for CSF (0), arterial
        # (1) and venous (2), Neumann (1) for capillaries (3)
        if (boundaries.array()[i] == SKULL):
            problem.momentum_boundary_markers[i] = DIRICHLET
            problem.continuity_boundary_markers[0][i] = DIRICHLET
            problem.continuity_boundary_markers[1][i] = DIRICHLET
            problem.continuity_boundary_markers[2][i] = DIRICHLET
            problem.continuity_boundary_markers[3][i] = NEUMANN
        # Boundary conditions on the ventricles for the different
        # compartments: Dirichlet (0) conditions for CSF (0) and
        # venous (2), Neumann (1) for the arterial (1) and
        # capillaries (3)
        elif (boundaries.array()[i] == VENTRICLES):
            problem.momentum_boundary_markers[i] = NEUMANN
            problem.continuity_boundary_markers[0][i] = DIRICHLET
            problem.continuity_boundary_markers[1][i] = NEUMANN
            problem.continuity_boundary_markers[2][i] = DIRICHLET
            problem.continuity_boundary_markers[3][i] = NEUMANN
        else:
            pass    

    # Set-up solver
    direct_solver = (solver_type == "direct")
    params = dict(dt=dt, theta=theta, T=T,  direct_solver=direct_solver)

    # Define storage for the displacement and the pressures
    prefix = "results_brain/nu_" + str(nu)\
             + "_formulationtype_" + formulation_type\
             + "_solvertype_" + solver_type 
    fileu = XDMFFile(prefix + "/u.pvd")
    filep = [XDMFFile(prefix + "/p%d.pvd" % (i+1)) for i in range(A)]

    fileu_pvd = File(prefix + "/pvd/u.pvd")
    filep_pvd = [File(prefix + "/pvd/p%d.pvd" % (i+1)) for i in range(A)]
    
    # Store mesh and markers in a bunch of formats
    filemesh = File(prefix+"/mesh.pvd")
    filemesh << mesh

    filemarkers = File(prefix+"/markers.pvd")
    filemarkers << markers
    
    fileboundaries = File(prefix+"/skull_ventricles.pvd")
    fileboundaries << boundaries

    if formulation_type == "standard":
        print("Solve with standard solver")

        # Initialize solver
        solver = MPETSolver(problem, params)

        # Set initial conditions: zero displacement at t = 0, and set
        # pressure initial conditions based on p_bar.
        VP = solver.up_.function_space()
        V = VP.sub(0).collapse()
        assign(solver.up_.sub(0), interpolate(Constant((0.0, 0.0, 0.0)), V))
        for i in range(A):
            Q = VP.sub(i+1).collapse()
            assign(solver.up_.sub(i+1), interpolate(problem.p_bar[i], Q))

        # Split and store initial solutions
        solver.up.assign(solver.up_)
        values = solver.up.split()
        u = values[0]
        p = values[1:]
        fileu.write(u, 0.0)
        for i in range(A):
            filep[i].write(p[i], 0.0)

        fileu_pvd << u
        for i in range(A):
            filep_pvd[i] << p[i]
                     
        # Solve away!
        print("Number of degrees of freedom = ", VP.dim())
        solutions = solver.solve()
        
        t_start = timemodule.time()
        for (up, t) in solutions:
            info("t = %g" % t)

            # Split and store solutions 
            values = up.split()
            u = values[0]
            p = values[1:]

            fileu.write(u, t)
            for i in range(A):
                filep[i].write(p[i], t)

            fileu_pvd << u
            for i in range(A):
                filep_pvd[i] << p[i]

            t_stop = timemodule.time()
        print("Solver time = %0.3g (s)" % (t_stop - t_start))
    else:
        pass
        
if __name__ == "__main__":

    import sys

    #nu = 0.497
    nu = 0.4999
    formulation_type = "standard"
    solver_type = "direct"
    
    # Read mesh and other mesh related input
    mesh, markers, boundaries = create_mesh()

    # Run simulation
    mpet_solve(mesh, markers, boundaries,
               M=20, theta=1.0, nu=nu,
               formulation_type=formulation_type,
               solver_type=solver_type)

    # Display some timings
    list_timings(TimingClear.keep, [TimingType.wall,])
