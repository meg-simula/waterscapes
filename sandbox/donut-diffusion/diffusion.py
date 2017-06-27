# Code example simulating diffusion of a concentration injected inside
# the brain, used for Rognes TEDxOslo 2017 slides.

"""
  c_t - D div (grad c) = f   in O 
                     c = 0   on dO
                  c(0) = 0   initial condition

Semi-discretization:

   <c_t, q> + <D grad c, grad q> - <D grad c * n, q>_{dO} = <f, q>,

Crank-Nicolson:

   <c - c^-/dt, q> + < D grad(c^m), grad q> - < D grad(c^m) * n, q> = <f^m, q>,

where 2 c^m = (c + c^-).

"""

#import numpy
import math
from dolfin import *
from mshr import *

def generate_mouse_donut_mesh():

    # Get as much output as possible by setting debug type log level.
    set_log_level(DEBUG)

    origin = Point(0.0, 0.0)
    r1 = 5.0 # Outer radius (mm)
    r2 = 1.0 # Inner radius  (mm)
    
    parenchyma = Circle(origin, r1)
    ventricles = Circle(origin, r2)
    
    geometry = parenchyma - ventricles
    dolfin.info(geometry, True)
    
    # Create mesh, N controls the resolution (N higher -> more cells)
    N = 20
    mesh = generate_mesh(geometry, N)
    
    plot(mesh, title="mesh")
    
    # Store mesh to h5 (good for further FEniCS input/output)
    file = HDF5File(mpi_comm_world(), "mouse_donut2D.h5", "w")
    file.write(mesh, "/mesh")
    file.close()

    # Store mesh to pvd/vtu (good for Paraview)
    file = File("mouse_donut2D.pvd")
    file << mesh

    return mesh

def main():

    #generate_mouse_donut_mesh()
    #exit()
    
    mesh = Mesh()
    hdf = HDF5File(mpi_comm_world(), "mouse_donut2D.h5", "r")
    hdf.read(mesh, "/mesh", False)
    plot(mesh)
    #plot(markers, interactive=True)

    # Variational formulation
    Q = FunctionSpace(mesh, "CG", 1)
    c = TrialFunction(Q)
    q = TestFunction(Q)
    
    c_ = Function(Q)
    
    dt = Constant(10.)
    t = Constant(dt)
    D = Constant(8.7e-4)
    
    bcs = [DirichletBC(Q, 0.33, "((x[0]*x[0] + x[1]*x[1]) > 2*2) && on_boundary"),]
#           DirichletBC(Q, 0.0, "((x[0]*x[0] + x[1]*x[1]) < 2*2) && on_boundary")]

    cm = 1./2*(c + c_)
    F = (c - c_)*q*dx() + dt*inner(D*grad(cm), grad(q))*dx()
    a, L = system(F)
    
    # Assemble left hand side matrix
    info("Assembling")
    A = assemble(a)
    b = Vector(mpi_comm_world(), Q.dim())
    
    # Step in time
    c = Function(Q)
    c.assign(c_)
    N = 6*120
    
    file = File("output/c.pvd")
    file << c
    
    if True:
        solver = LUSolver(A)
        solver.parameters["reuse_factorization"] = True
    else:
        solver = PETScKrylovSolver("cg", "amg")
        solver.set_operator(A)
        #solver.parameters["monitor_convergence"] = True
    
    for i in range(N):
        print "Solving %d" % i
        
        assemble(L, tensor=b)
        
        # Apply boundary conditions
        for bc in bcs:
            bc.apply(A, b)

        # Solve linear system
        solver.solve(c.vector(), b)
        
        # Update previous solution
        t.assign((float(t) + float(dt)))
        c_.assign(c)
    
        plot(c, key="c")
        file << c
    
    list_timings(TimingClear_keep, [TimingType_wall])
    print "Success, all done!"
    interactive()

if __name__ == "__main__":
    main()
