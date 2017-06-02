
from mpet import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
import time

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
#parameters["form_compiler"]["representation"] = "uflacs"

class NaiveSquare(MPET):

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
        self.mesh = UnitSquareMesh(n, n)

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

    def initial_conditions(self, t):
        AA = self.params.AA
        u0 = Constant((0.0, 0.0))
        p0 = [Constant(0.0) for i in range(AA)]
        return u0, p0

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."
        AA = self.params.AA
        u0 = [(Constant((0.0, 0.0)), self.allboundary),]
        p0 = [((i+1), Constant(0.0), self.allboundary) for i in range(AA)]
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        u0 = []
        p0 = []
        return u0, p0

    def f(self, t):
        ff = Expression(("t", "0.0"), t=t)
        return ff

    def g(self, t):
        gg = [Constant(0.0), Constant(0.0)]
        return gg
    
    def nullspace(self):
      
        # No null space
        null = []
        return null

def main(N=16, M=4):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 1.0
    dt = float(T)/M
    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    Ks = (1.0, 1.0)
    G = (1.0, 1.0)

    E = 1.0
    nu = 0.35
    Incompressible = False

    # Create problem set-up
    problem = NaiveSquare(dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                               Ks=Ks, G=G, Incompressible=Incompressible,
                               E=E, nu=nu))

    # Create solver"krylov_solver"]["report"] = True
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "direct_solver": False, "krylov_solver": {"monitor_convergence":True}})

    # Solve
    timer = Timer("A: MPET solve")
    solutions = solver.solve()
    for (U, t) in solutions:
        print "Solving at t = %g" % t
        pass
    timer.stop()

    u = U.split(deepcopy=True)[0]
    p = U.split(deepcopy=True)[1:]

    plot(u, title="u at T = %g" % t)
    plot(p[0], title="p0 at T = %g" % t)
    plot(p[1], title="p1 at T = %g" % t, interactive=True)

    # Output timings
    list_timings(TimingClear_keep, [TimingType_wall])

if __name__ == "__main__":

    main()
