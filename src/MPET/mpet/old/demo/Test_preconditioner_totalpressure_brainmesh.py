
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

#set_log_level(30)
#parameters["form_compiler"]["representation"] = "uflacs"
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# pylab.rc('text', usetex=True)
# pylab.rc('font', family='serif')
            
class FirstTest(MPET):

    """2 networks:
    -1st: extracellular CSF
    -2nd: capillary
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
        self.facet_domains = FacetFunction("size_t", self.mesh)
        self.facet_domains.set_all(0)

        # Mark all exterior boundary facets as 1
        # Note: The CompiledSubDomain class gives faster code than
        # Python subclassing
        #self.allboundary = AllBoundary()
        self.allboundary = CompiledSubDomain("on_boundary")
        self.allboundary.mark(self.facet_domains, 1) #!!!

        gdim = self.mesh.geometry().dim()
        self.cell_domains = MeshFunction("size_t", self.mesh, gdim,
                                         self.mesh.domains())
        print "boundaries defined"


    def exact_solutions(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        u = Expression(("0.0", "0.0", "0.0"),
                        domain=self.mesh, degree=3)
        # p0 = Expression("A0*sin(2.0*pi*t)+C0",
        #                 A0 = 666.611825, C0 = 1999.835475, t=t,domain=self.mesh, degree=3)
        p0 = Expression("A0*sin(2.0*pi*t)+C0",
                        A0 = 666.611825, C0 = 1000, t=t,domain=self.mesh, degree=3)
        p1 = Expression("C1",
                        C1=650.0, domain=self.mesh, degree=3)
        p = [p0, p1]
        return u, p
    
    
    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L
        n = self.period
        nu = self.params.nu
        E = self.params.E

        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
        
        alphas = self.params.alphas
        
        uex, pex = self.exact_solutions(t)
        
        x = SpatialCoordinate(self.mesh)
        
        # u0 = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", "sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"),
        #                 n=n, L=L, t=t, domain=self.mesh, degree=5)
        
        # p0 = 1999.835475
        p0 = pex[0]
        p1 = pex[1]
        divu = 0.0
        pt = lmbda * divu - alphas[0] * p0 - alphas[1] * p1
        
        p = [pt, pex[0], pex[1]]

        return uex, p

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions(t)
        
        u0 = [(uex, self.allboundary)]

        p0 = [((2), pex[0], self.allboundary), ((3), pex[1], self.allboundary)]

        u0 = []
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        u0 = []
        p0 = []
        return u0, p0

    def f(self, t):
    
        ff = Expression(("0.0", "0.0", "0.0"),
                        domain=self.mesh, degree=1)
        
        return ff
    
    def g(self, t):

        AA = self.params.AA

        gg = [0]*AA
        
        gg[0] = Expression("0.0", domain=self.mesh, degree=1)
        gg[1] = Expression("0.0", domain=self.mesh, degree=1)
        
        return gg
    

    
    def nullspace(self):
      
        # No null space
        null = True
        return null


def single_run():
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 1.0
    M = 40.0
    dt = float(T/M)
    T = 1.0
    # dt = T

    L = 1.0
    Q = 1.0/(4.5e-10)
    AA = 2
    alphas = (1.0, 1.0)
    Ks = (1.573e-5, 0.03745)
    G = (1.0e-13,1.0e-13)

    E = 584

    nu = 0.35

    Incompressible = False

    mesh = "../../meshes/whitegray.xml.gz"
    mesh = Mesh(mesh)
    #mesh = UnitCubeMesh(10,10,10)
    print "lmbda = ", nu*E/((1.0-2.0*nu)*(1.0+nu))
    print "mu = ", E/(2.0*(1+0+nu))

    print "Ks = ", Ks
    # Create problem set-up 
    problem = FirstTest(dict(mesh_file=mesh, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    print problem.params
    
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 1.0, "u_degree":1, "direct_solver":False, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})


    # Solve
    fileu = File("results/u_brain.pvd")
    filep0 = File("results/p0_brain.pvd")
    filep1 = File("results/p1_brain.pvd")
    filep2 = File("results/p2_brain.pvd")
    solutions = solver.solve_totalpressure()
    for (U, t) in solutions:
        u = U.split(deepcopy=True)[0]
        p = U.split(deepcopy=True)[1:]
        
        # The adjoint should be in the solver?
        
        # p_CSF = Function(Ks[0])
        # p_CSF.assign(p[0])
        # J = Functional(inner(grad(p[0]), grad(p[0]))*dx*dt[FINISH_TIME])
        
        

        fileu<<u
        filep0<<p[0]
        filep1<<p[1]
        filep2<<p[2]

    print "*"*200

if __name__ == "__main__":

    single_run()

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
