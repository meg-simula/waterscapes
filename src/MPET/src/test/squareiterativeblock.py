"""run with python prova.py "$(<params.txt)"
use cbcpost and set_parse_command_line_arguments(True)
"""
from mpet import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
from dolfin import *
# from dolfin_adjoint import *
# set_parse_command_line_arguments(True)

set_log_level(PROGRESS)

class Ventricles(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[0]**2/(40.0**2)+(x[1]+30.0)**2/(60.0**2)+(x[2]-2.0)**2/(30.0**2) < 1.0)

class Skull(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[0]**2/(65.0**2)+(x[1])**2/(65.0**2)+(x[2])**2/(65.0**2) > 1.0)

class AllBoundary(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return on_boundary


# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["representation"] = "uflacs"

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

        # self.mesh = UnitCubeMesh(6,6,6)

        # Create boundary markers and initialize all facets to +
        self.facet_domains = FacetFunction("size_t", self.mesh)
        self.facet_domains.set_all(0)

        self.allboundary = AllBoundary()
        self.allboundary.mark(self.facet_domains, 1)
        
        # self.ventricles = Ventricles()
        # self.ventricles.mark(self.facet_domains, 2) #!!!
        # 
        # self.skull = Skull()
        # self.skull.mark(self.facet_domains, 3)
        
        gdim = self.mesh.geometry().dim()
        self.cell_domains = MeshFunction("size_t", self.mesh, gdim,
                                         self.mesh.domains())

        # Attach domains to measures for convenience
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_domains)


    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L

        u0 = Constant((1.0, 0.0))
        p0 = [Constant(0.0) for i in range(AA)]
        return u0, p0


    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L

        # Value, boundary subdomain
        u0 = [(Expression(("1.0","0.0"), domain=self.mesh, degree=5, t=t, L=L), self.allboundary)]
        p0 = [((1), Expression("a*sin(pi/30.0 * t)", domain=self.mesh, degree=5, t=t, L=L, a=1.0), self.allboundary),\
              ((2), Expression("a*sin(pi/30.0 * t)", domain=self.mesh, degree=5, t=t, L=L, a=1.0), self.allboundary) ]

        # u0 = []
        # p0 = [((1), Expression("a*sin(pi/30.0 * t)", domain=self.mesh, degree=5, t=t, L=L, a=1.0), self.allboundary)]
        return u0, p0

    def neumann_conditions(self, t, t_dt, theta):
        
        # u0 =  [(Expression("a*sin(pi/30.0 * t)", domain=self.mesh, degree=5, t=t_dt, L=L, a=1.0), 1)]
        u0 = []
        p0 = []
        return u0, p0

    def f(self, t):

        ff = Constant((0.0, 0.0))
        return ff


    def g(self, t):

        AA = self.params.AA

        gg = [0]*AA
        gg[0] = Constant(0.0)
        gg[1] = Constant(0.0)
        return gg

    def exact_solutions(self, t):
        pass
    
    def nullspace(self):
        
        # Nullspace:

        # Rigid motions: 3 translations + 3 rotations
        # null = [Constant((1, 0)), Constant((0, 1)), Expression(('x[1]', '-x[0]'))]

        # RM + Blood Pressure defined up to a constant
        # null = [Constant((1, 0, 0, 0, 0)), Constant((0, 1, 0, 0, 0)), Constant((0, 0, 1, 0, 0)),\
        #              Expression(('0', 'x[2]', '-x[1]', '0', '0')), Expression(('-x[2]', '0', 'x[0]', '0', '0')), Expression(('x[1]', '-x[0]', '0', '0', '0')),\
        #              Constant((0, 0, 0, 0, 1))]

        # Blood Pressure defined up to a constant
        # null = [Constant((0, 0, 0, 0, 1))]
        
        # No null space
        null = []
        
        return null

    

def single_run():


    # parameters["adjoint"]["stop_annotating"] = True
    # Create problem set-up
    T = 1.0
    dt = 0.01
    # dt = 1.0
    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    # Ks = (Permeability(), )
    Ks = (1.0, 1.0)
    G = (1.0, 1.0)
    
    E = 1.0
    nu = 0.3
    
    Incompressible = False
    N = 32
    mesh = UnitSquareMesh(N, N)
    # Create problem set-up
    print "problem set up"
    problem = FirstTest(dict(mesh_file=mesh, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    # Create solver
    print "solver..."
    solver = SimpleSolver(problem, {"dt": dt, "theta":1.0, "T": T, "direct_solver": False, "testing":True, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                    "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-6, "divergence_limit": 1.e10}})

    # Solve
    solutions = solver.solve_block()
    for (u,p0,p1, t) in solutions:
        pass
    
    # u = U.split(deepcopy=True)[0]
    # p = U.split(deepcopy=True)[1:]
    
    # plot(p[0], problem.mesh)
    # interactive()

def check_boundaries():
    problem = FirstTest()
    File("boundaries.pvd")<< problem.facet_domains


single_run()
# check_boundaries()