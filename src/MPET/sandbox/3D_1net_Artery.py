from dolfin import *
from mpet import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
import time
import pylab
import resource

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
        
        print "L = ", self.params.L
        self.facet_domains = FacetFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        
        self.allboundary = CompiledSubDomain("on_boundary")        
        self.allboundary.mark(self.facet_domains, 0)

        self.outside = CompiledSubDomain("on_boundary && (x[0] == -L || x[0] == L ||\
                                                          x[1] == -L || x[1] == L ||\
                                                          x[2] == 0.0 || x[2] == 2*L)", L=self.params.L)
        
        self.inside = CompiledSubDomain("on_boundary && (x[0]*x[0] + x[1]*x[1] <= L/10*L/10 + 2 * DOLFIN_EPS)", L=self.params.L)        

        self.inside.mark(self.facet_domains, 1)
        self.outside.mark(self.facet_domains, 2)

        File("subdomains_artery.pvd") << self.facet_domains

    def exact_solutions(self, t):

        u = Constant((0.0, 0.0, 0.0))
        p = [Constant(0.0),]

        return u, p
    
    
    def initial_conditions(self, t):
     
        uex, pex = self.exact_solutions(t)
 
        return uex, pex

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."
                
        p0 = [((1), Constant(0.0), self.outside)]
        u0 = [(Expression(("A*x[0]/r*sin(2*pi*t)", "A*x[1]/r*sin(2*pi*t)","0.0"), domain=self.mesh, A=0.1 * (self.params.L)/10, r = (self.params.L)/10, t=t, degree=3), self.inside)]
        # u0 = [(Expression(("t", "0.0","0.0"), domain=self.mesh, t=t, degree=3), self.inside),]

        
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        
        u0 = []
        p0 = []
        
        return u0, p0

    def f(self, t):
        
        #Let's try with Womersley force
        #Amplitude is 40mmhg that is approximately 5000 Pa
        ff = Constant((0.0,0.0,0.0))
        return ff
    
    def g(self, t):

        AA = self.params.AA
        gg = [0]*AA     
        gg[0] = Constant(0.0)
        
        return gg
    
    def nullspace(self):
      
        # No null space
        null = False
        return null


def single_run():

    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    N= 64
    M = 50.0
    dt = float(1.0/M)
    T = 3.0
    # T = dt
    L = 1.0 
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
        
    mesh = ("../../meshes/Artery_coarse.xml")
    mesh = refine(Mesh(mesh))
    # mesh = UnitCubeMesh(10,10,10)
    # mesh = refine(mesh)
    # File("Artery_refinedmesh.pvd")<<mesh
    
    # Create problem set-up 
    problem = FirstTest(dict(mesh_file=mesh, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))
    
    foldername = "3D_1net_Artery_coarse_refined" + "_nullspace_%s" %problem.nullspace() +\
                 "_E_" + "%04.03e" %E + "_K_" + "%04.03e" %Ks[0] + "_Q" + "%04.03e" %Q
    
    print problem.params

    Vf = VectorFunctionSpace(problem.mesh, "CG", 1)
    Pf = FunctionSpace(problem.mesh, "CG", 1)
    vfinterp = Function(Vf)
    
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 0.5, "direct_solver":True, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-10, "absolute_tolerance": 1.e-6, "divergence_limit": 1.e10}})

    # Solve
    filenameu = "results/" + foldername + "/u_2D.xdmf"
    filenamep0 = "results/" + foldername + "/p0_2D.xdmf"
    filenamevf = "results/" + foldername + "/vf_2D.xdmf"
    
    fileu = XDMFFile(mpi_comm_world(), filenameu)
    filep0 = XDMFFile(mpi_comm_world(), filenamep0)
    filevf = XDMFFile(mpi_comm_world(), filenamevf)

    # filenameu = "results/" + foldername + "/u_2D.pvd"
    # filenamep0 = "results/" + foldername + "/p0_2D.pvd"
    # filenamef = "results/" + foldername + "/f_2D.pvd"
    
    # fileu = File(filenameu)
    # filep0 = File(filenamep0)
    # filef = File(filenamef)

    # solutions = solver.solve_totalpressure()
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print 'Memory usage: %s (kb)' % (mem)
    solutions = solver.solve_symmetric()
    
    for (U, t) in solutions:
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print 'Memory usage: %s (kb)' % (mem)

        u = U.split(deepcopy=True)[0]
        p = U.split(deepcopy=True)[1:AA+1]
        # plot(f, mesh=problem.mesh)
        vf = -Ks[0] * grad(p[0])
        vf = project(vf, Vf)
        # fileu << u
        # filep0 << p[0]
        # filevf << vf
        
        fileu.write(u)
        filep0.write(p[0])
        filevf.write(vf)
                

if __name__ == "__main__":

    single_run()

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
