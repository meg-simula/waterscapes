from dolfin import *
from mpet import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
import time
import pylab

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
            
class FirstTest(MPET):

    """
    First setup considering a 2D mesh and 1 network.
    Only neumann conditions are imposed on the displacement.
    Parameters taken from Tully and Ventikos 2010"""

    def __init__(self, params=None):

        MPET.__init__(self, params)

        # Create mesh
        m = self.params.mesh_file
        if isinstance(m, Mesh):
            self.mesh = m
        else:
            self.mesh = Mesh(m)

        self.period = 2*math.pi
        
        # Create boundary markers and initialize all facets to +
        self.facet_domains = FacetFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)

        
        self.allboundary = CompiledSubDomain("on_boundary")        
        self.left = CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS")        
        self.right = CompiledSubDomain("on_boundary && x[0] > L - DOLFIN_EPS", L = self.params.L)        
        
        self.allboundary.mark(self.facet_domains, 1)
        self.left.mark(self.facet_domains, 2)
        self.right.mark(self.facet_domains, 3)


    def exact_solutions(self, t):

        u = Constant((0.0, 0.0))
        p = [Constant(0.0), ]

        return u, p
    
    
    def initial_conditions(self, t):
     
        uex, pex = self.exact_solutions(t)
 
        return uex, pex

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        uex, pex = self.exact_solutions(t)
                
        p0 = [((1), pex[0], self.left),((1), pex[0], self.right),]
        u0 = [(uex, self.left), (uex, self.right)]
        
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        
        u0 = []
        p0 = []
        
        return u0, p0

    def f(self, t):
        
        #Let's try with Womersley force
        #Amplitude is 40mmhg that is approximately 5000 Pa
        Gaussian = Expression("exp(-pow((x[0]-mu),2)/(2*pow(sigma,2)))*A*sin(2*pi*t)*(x[1] - L)", L = self.params.L, A = 5000, t=t, mu=1.0, sigma=0.1, degree=3, domain = self.mesh)
        ff = grad(Gaussian)
        
        return ff
    
    def g(self, t):

        AA = self.params.AA
        gg = [0]*AA     
        gg[0] = Constant(0.0)
        
        return gg
    

    
    def nullspace(self):
      
        # No null space
        null = False
        # null = True
        return null


def single_run():

    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    N= 64
    M = 50.0
    dt = float(1.0/M)
    T = 3.0
    # T = dt
    L = 2.0 
    Q = 1.0/(4.5e-10)
    AA = 1
    alphas = (1.0,)
    Ks = (1.5e-5,)
    G = (1.0e-13,)

    E = 584.0
    nu = 0.35

    Incompressible = False

    print "lmbda = ", nu*E/((1.0-2.0*nu)*(1.0+nu))
    print "mu = ", E/(2.0*(1.0+nu))

    print "Ks = ", Ks
 
    #Loading mesh from file
        
    mesh = RectangleMesh(Point(0.0,0.0), Point(L,L), 64, 64, "crossed")
    # mesh = UnitSquareMesh(N,N)
    
    # Create problem set-up 
    problem = FirstTest(dict(mesh_file=mesh, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    problem.params.E = Expression("500 + exp(-pow((x[0]-mu),2)/(2*pow(sigma,2)))*A", A = 5000, mu=1.0, sigma=0.1, degree=3, domain = problem.mesh)
 
    plot(problem.params.E, mesh=problem.mesh)
    foldername = "2D_1Net_PulsatileBloodWomersley_DirichletLeftRight" + "_nullspace_%s" %problem.nullspace() +\
                 "_Emaxgaussian_5000_K_" + "%04.03e" %Ks[0] + "_Q" + "%04.03e" %Q
    
    print problem.params

    Vf = VectorFunctionSpace(problem.mesh, "CG", 1)
    Pf = FunctionSpace(problem.mesh, "CG", 1)
    vfinterp = Function(Vf)
    
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 1.0, "direct_solver":True, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-10, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})


    # Solve
    filenameu = "results/" + foldername + "/u_2D.xdmf"
    filenamep0 = "results/" + foldername + "/p0_2D.xdmf"
    filenamevf = "results/" + foldername + "/vf_2D.xdmf"
    filenamef = "results/" + foldername + "/f_2D.xdmf"
    
    fileu = XDMFFile(mpi_comm_world(), filenameu)
    filep0 = XDMFFile(mpi_comm_world(), filenamep0)
    filevf = XDMFFile(mpi_comm_world(), filenamevf)
    filef = XDMFFile(mpi_comm_world(), filenamef)

    # filenameu = "results/" + foldername + "/u_2D.pvd"
    # filenamep0 = "results/" + foldername + "/p0_2D.pvd"
    # filenamef = "results/" + foldername + "/f_2D.pvd"
    
    # fileu = File(filenameu)
    # filep0 = File(filenamep0)
    # filef = File(filenamef)

    # solutions = solver.solve_totalpressure()
    solutions = solver.solve_symmetric()
    
    for (U, t) in solutions:
        u = U.split(deepcopy=True)[0]
        p = U.split(deepcopy=True)[1:AA+1]
        f = Expression("exp(-pow((x[0]-mu),2)/(2*pow(sigma,2)))*A*sin(2*pi*t)*(x[1] - L)",\
                               L = problem.params.L, A = 5000, t=t, mu=1.0, sigma=0.1, degree=3, domain = problem.mesh)
        # plot(f, mesh=problem.mesh)
        vf = -Ks[0] * grad(p[0])
        vf = project(vf, Vf)
        ff = interpolate(f, Pf)
        # fileu << u
        # filep0 << p[0]
        # filevf << vf
        
        fileu.write(u)
        filep0.write(p[0])
        filevf.write(vf)
        filef.write(ff)
                

if __name__ == "__main__":

    single_run()

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
