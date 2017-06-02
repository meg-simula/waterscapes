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
    First setup considering a 1D mesh and 1 metwork.
    Only neumann conditions are imposed on the displacement.
    Rigid motions are taken into account using lagrange multipliers.
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

        self.rV = 30.0 # Ventricles 
        self.rS = 70.0 # Skull

        self.allboundary = CompiledSubDomain("on_boundary")
        self.ventricles = CompiledSubDomain("on_boundary & x[0]*x[0]+x[1]*x[1] <= rV*rV + 1e-9", rV = self.rV)
        self.skull = CompiledSubDomain("on_boundary & x[0]*x[0]+x[1]*x[1] >= rV*rV", rV=self.rV)

        
        self.allboundary.mark(self.facet_domains, 1)
        self.ventricles.mark(self.facet_domains, 2) 
        self.skull.mark(self.facet_domains, 3) 
       
    def exact_solutions(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        
       
        u = Constant((0.0, 0.0))
        # p0 = Expression("(A0 - 13.0/(L) * x[0]) * sin(2*pi*t)",
        #                 A0 = 666.611825, L = L, t = t, domain=self.mesh, degree=3)
                        # A0 = 0.0, L = L, t = t, domain=self.meshhttps://fenicsproject.org/documentation/dolfin/dev/python/programmers-reference/compilemodules/subdomains/CompiledSubDomain.html, degree=3)
        # p0 = Expression("A0*sin(2.0*pi*t)",
                        # A0 = 666.611825, t=t,domain=self.mesh, degree=3)
        # p0 = Expression("A*(1.0 - x[0]/L) + B", degree=1, A=13.0, B=0.0, L=L)
        p0 = Constant(0.0)
        pt = Constant(0.0)
        p = [pt, p0,]
        return u, p
    
    
    def initial_conditions(self, t):
       
        uex, pex = self.exact_solutions(t)
     
        return uex, pex

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        uex, pex = self.exact_solutions(t)
                
        p0 = [((2), Expression("A0*sin(2.0*pi*t)", A0 = 500, t=t, domain=self.mesh, degree=3), self.skull),]
        #u0 = [(uex, self.skull), (uex, self.ventricles)]
        u0 = [(uex, self.skull)]

        # u0 = []

        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        
        u0 = []
        p0 = []

        return u0, p0

    def f(self, t):
        
        ff = Constant((0.0,0.0))
        return ff
    
    def g(self, t):

        AA = self.params.AA

        gg = [0]*AA
        
        gg[0] = Constant(0.0)
        
        return gg
    

    
    def nullspace(self):
      
        # No null space
        # null = False
        null = False
        return null


def single_run():

    "N is the mesh size, M the number of time steps."

    # foldername = "2D_1Net_FixedGradP_Neumann_refined"
    # foldername = "2D_1Net_FixedGradP_FixedBrain_K1.5e-5_totalpressure_"
    # Specify discretization parameters
    
    M = 40.0
    dt = float(1.0/M)
    T = 10.0
    # T = 4*dt
    L = 70.0 
    Q = 1.0/(4.5e-10)
    AA = 1
    alphas = (1.0,)
    Ks = (1.5e-5,)
    G = (1.0e-13,)

    E = 5000.0
    nu = 0.49999
    
    Incompressible = False

    print "lmbda = ", nu*E/((1.0-2.0*nu)*(1.0+nu))
    print "mu = ", E/(2.0*(1.0+nu))

    print "Ks = ", Ks
 
    #Loading mesh from file
    mesh = Mesh("../../meshes/Donut_coarse.xml")

    # Create problem set-up 
    problem = FirstTest(dict(mesh_file=mesh, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))
    foldername = "2D_1Net_PulsatilePressure_FixedBrain_NoFluxOnVentricles" + "_nullspace_%s" %problem.nullspace() +\
                 "_E_" + "%04.03e" %E + "_nu_" + "%04.03e" %nu +"_K_" + "%04.03e" %Ks[0] + "_Q" + "%04.03e" %Q + "_Donut_coarse_totalpressure"

    # plot(problem.mesh)
    # plot(problem.facet_domains)    
    interactive()
    print problem.params

    Vf = VectorFunctionSpace(problem.mesh, "CG", 1)
    vfinterp = Function(Vf)
    
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 1.0, "direct_solver":False, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})


    # Solve
    # filenameu = "results/" + foldername + "/u.xdmf"
    # filenamep1 = "results/" + foldername + "/p1.xdmf"
    # filenamevf = "results/" + foldername + "/vf.pvd"
    
    # fileu = XDMFFile(mpi_comm_world(), filenameu)
    # filep1 = XDMFFile(mpi_comm_world(), filenamep1)
    # filevf = XDMFFile(mpi_comm_world(), filenamevf)


    filenameu = "results/" + foldername + "/u_2D.pvd"
    filenamep0 = "results/" + foldername + "/p0_2D.pvd"
    filenamevf = "results/" + foldername + "/vf_2D.pvd"
    
    fileu = File(filenameu)
    filep0 = File(filenamep0)
    filevf = File(filenamevf)


    solutions = solver.solve_totalpressure()
    # solutions = solver.solve_symmetric()
    
    for (U, t) in solutions:
        u = U.split(deepcopy=True)[0]
        p = U.split(deepcopy=True)[1:AA+2]
        vf = -Ks[0] * grad(p[1])
        vf = project(vf, Vf)
        # The adjoint should be in the solver?
        
        # p_CSF = Function(Ks[0])
        # p_CSF.assign(p[0])
        # J = Functional(inner(grad(p[0]), grad(p[0]))*dx*dt[FINISH_TIME])
        
        fileu << u
        filep0 << p[0]
        filevf << vf    

        #fileu.write(u)
        #filep1.write(p[1])
        #filevf.write(vf)
        # plot(u, key="u", title="displacement", mesh=mesh)
        # plot(p[0], key="p", title="pressure", mesh=mesh)
        # plot(vf, key="vf", title="flow velocity", mesh=mesh)
    # interactive()
        
    print "*"*200

if __name__ == "__main__":

    single_run()

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
