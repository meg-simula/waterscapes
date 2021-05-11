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
    Rigid motions are taken into account using Lagrange multipliers.
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
        p0 = Constant(0.0)

        p = [p0, p0]
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
        
        return uex, pex

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions(t)
        
        
        u0 = [(uex, self.skull),]        
        # p0 = [((1), Expression("A0*sin(2.0*pi*t)", A0 = 13.0, t=t, domain=self.mesh, degree=3), self.ventricles), ((1), Constant(0.0), self.skull)]
        p0 = [((1), Expression("A0*sin(2.0*pi*t)",A0 = 13.0, t=t, domain=self.mesh, degree=3), self.skull),]
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        
        pex1 = Expression("A0*sin(2.0*pi*t)", A0 = 13.0, t=t1, domain=self.mesh, degree=3)
        pex0 = Expression("A0*sin(2.0*pi*t)", A0 = 13.0, t=t0, domain=self.mesh, degree=3)
        
        d = 2
        I = Identity(d)        
        u0 = [(-pex1*I, 2)]
        # u0 = []
        # p0 = [(Constant((0.0,0.0)), 0, 3),]
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
        null = False
        # null = True
        return null


def single_run():

    "N is the mesh size, M the number of time steps."


    # Specify discretization parameters
    M = 50.0
    dt = float(1.0/M)
    T = 4.0
    # T = 4*dt
    L = 70.0 
    Q = 1.0/(4.5e-10)
    # Q = 1.0
    AA = 2
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
    mesh=Mesh()
    filename = "../../meshes/2D_brain_refined7.xdmf"
    f = XDMFFile(mpi_comm_world(), filename)
    f.read(mesh, True)
    # mesh = refine(mesh)
    # Create problem set-up 
    problem = FirstTest(dict(mesh_file=mesh, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    foldername = "2D_1Net_PulsatilePressureGradient_FixedSkull" + "_nullspace = %s" %problem.nullspace() +\
                 "_K_" + "%04.03e" %Ks[0] + "_Q" + "%04.03e" %Q + "_refined7"
    
    print problem.params

    Vf = VectorFunctionSpace(problem.mesh, "CG", 1)
    vfinterp = Function(Vf)
    
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 0.5, "direct_solver":True, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-10, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})


    # Solve
    # filenameu = "results/" + foldername + "/u_2D.xdmf"
    # filenamep0 = "results/" + foldername + "/p0_2D.xdmf"
    # filenamevf = "results/" + foldername + "/vf_2D.xdmf"
    # 
    # fileu = XDMFFile(mpi_comm_world(), filenameu)
    # filep0 = XDMFFile(mpi_comm_world(), filenamep0)
    # filevf = XDMFFile(mpi_comm_world(), filenamevf)

    filenameu = "results/" + foldername + "/u_2D.pvd"
    filenamep0 = "results/" + foldername + "/p0_2D.pvd"
    filenamevf = "results/" + foldername + "/vf_2D.pvd"
    
    fileu = File(filenameu)
    filep0 = File(filenamep0)
    filevf = File(filenamevf)

    # solutions = solver.solve_totalpressure()
    solutions = solver.solve_symmetric()
    
    for (U, t) in solutions:
        u = U.split(deepcopy=True)[0]
        p = U.split(deepcopy=True)[1:AA+1]
        vf = -Ks[0] * grad(p[0])
        vf = project(vf, Vf)

        fileu << u
        filep0 << p[0]
        filevf << vf
        
        plot(u, key="u", title="displacement", mesh=mesh)
        

if __name__ == "__main__":

    single_run()

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
