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

class MyExpression(Expression):
   
    def eval(self, v, x):

        if (x[1] <= 0.25 - 1e-9 or x[1] >= 0.75 + 1e-9):
            v[0] = 1.0
        else:
            v[0] = 1.0e-8
    
    
class FirstTest(MPET):

    """2 networks:
    -1st: extracellular CSF
    -2nd: capillary
    Parameters taken from Tully and Ventikos 2010"""

    def __init__(self, params=None):

        MPET.__init__(self, params)

        # Create mesh
        x0 = Point(0.0, 0.0)
        x1 = Point(self.params.L, self.params.L)
        n = self.params.N
        self.mesh = UnitSquareMesh(n, n, "crossed")
        # self.mesh = 
        # Create boundary markers and initialize all facets to +
        self.facet_domains = FacetFunction("size_t", self.mesh)
        self.facet_domains.set_all(0)

        self.left = CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS && (x[1] > 0.0 || x[1] < 1.0)")
        self.left.mark(self.facet_domains, 1)

        self.right = CompiledSubDomain("on_boundary && x[0] > 1.0 - DOLFIN_EPS && (x[1] > 0.0 || x[1] < 1.0)")
        self.right.mark(self.facet_domains, 2)
        
        self.top = CompiledSubDomain("on_boundary && x[1] > 1.0 - DOLFIN_EPS && (x[0] >= 0.0 || x[0] <= 1.0)")
        self.top.mark(self.facet_domains, 3)
        
        # self.bottom= CompiledSubDomain("on_boundary && x[1] < DOLFIN_EPS && (x[0] >= 0.0 || x[0] <= 1.0)")
        self.bottom= CompiledSubDomain("on_boundary && x[1] < DOLFIN_EPS")
        self.bottom.mark(self.facet_domains, 4)

        
    def initial_conditions(self, t):

        u0 = Constant((0.0,0.0))
        p0 = [Constant(0.0)]

        return u0, p0

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        u0 = [( Constant((0.0,0.0)), self.left), ( Constant((0.0,0.0)), self.right), ( Constant((0.0,0.0)), self.top)]
        p0 = [((1), Constant(0.0), self.bottom),]

        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        u0 = [(Constant(1.0), 4),]
        p0 = []
        return u0, p0

    def f(self, t):

        ff = Constant((0.0,0.0))
        
        return ff

    def g(self, t):

        gg = [Constant(0.0)]
        return gg

    def nullspace(self):
      
        # No null space
        null = False
        return null

def single_run(N=64, M=64):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    dt = 1.0
    T = dt
    L = 1.0
    Q = 1.0
    AA = 1
    alphas = (1.0, )
    
    Ks = (MyExpression(degree=0), )
    G = (1.0, )

    E = 5.0/2.0
    nu = 0.25
    
    print "lmbda = ", nu*E/((1.0-2.0*nu)*(1.0+nu))
    print "mu = ", E/(2.0*(1.0+nu))
    Incompressible = True

    # Create problem set-up
    problem = FirstTest(dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    DG = FunctionSpace(problem.mesh, "DG", 0)
    KsDG = interpolate(MyExpression(), DG)
    problem.params.Ks = (KsDG, )
    plot(KsDG, mesh=problem.mesh)
    interactive()
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta":0.5, "direct_solver": True, "u_degree": 1, "p_degree":1})

    foldername = "RodrigoTest_SymmetricSolver_P1P1_deltat_1"
    filenameu = "results/" + foldername + "/u_2D.xdmf"
    filenamep0 = "results/" + foldername + "/p0_2D.xdmf"
    
    fileu = XDMFFile(mpi_comm_world(), filenameu)
    filep0 = XDMFFile(mpi_comm_world(), filenamep0)

         
    # Solve
    solutions = solver.solve_symmetric()
    for (U, t) in solutions:
        u = U.split(deepcopy=True)[0]
        p = U.split(deepcopy=True)[1:]
        fileu.write(u)
        filep0.write(p[0])
        plot(u, key="u", mesh=problem.mesh)
        plot(p[0], key="p", mesh=problem.mesh)
        
        print p[0](0.75,1.0)
    File("results/" + foldername + "/Ks.pvd") << problem.params.Ks[0]
    
if __name__ == "__main__":

    # Just test a single run
    single_run(N=32, M=32)

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    # main()
