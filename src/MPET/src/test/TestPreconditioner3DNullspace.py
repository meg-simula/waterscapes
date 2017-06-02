from dolfin import *
from mpet import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
import time
import pylab
from csf_pressure import csf_pressure
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
        self.rS = 100.0 # Skull

        self.allboundary = CompiledSubDomain("on_boundary")

        
        self.allboundary.mark(self.facet_domains, 1)
       
    def CSF_pressure(self, t):
        I_length, c, s = csf_pressure()
        expr = ""
        for i in range(len(c)):
            expr += str(c[i]) + "*cos(2*pi/I*" + str(i) + "*time)"
            expr += "+"
            expr += str(s[i]) + "*sin(2*pi/I*" + str(i) + "*time)"
            expr += "+"
        expr = expr[0:-1]
        p = Expression(expr, time=t, I = I_length, domain=self.mesh, degree=2) 
        #p = Expression("c*sin(t)", c=c[0], t=t, domain=self.mesh, degree=2)
        return p

    def exact_solutions(self, t):

        u = Constant((0.0, 0.0, 0.0))
        p = [Constant(0.0), ]

        return u, p
    
    
    def initial_conditions(self, t):
     
        uex, pex = self.exact_solutions(t)
 
        return uex, pex

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        uex, pex = self.exact_solutions(t)
                
        p0 = [((1), self.CSF_pressure(t), self.allboundary)]
        #u0 = [(uex, self.skull), (uex, self.ventricles)]
        #u0 = [(uex, self.skull)]
        u0 = []
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        
        #we impose no flux on the ventricles, so no need to add anything here!
        stress0 = self.CSF_pressure(t0)
        stress1 = self.CSF_pressure(t1)
        u0 = [(theta * stress1 + (1.0 - theta) * stress0, 1)]
        p0 = []
        
        return u0, p0

    def robin_conditions(self, t0, t1, theta):
        
        #for the moment we just pass the index of the boundary where we want to impose robin conditions
        #at the moment only 
        u0 = []
        
        return u0
    
    def f(self, t):
        
        ff = Constant((0.0,0.0,0.0))
        
        return ff
    
    def g(self, t):

        AA = self.params.AA
        gg = [0]*AA     
        gg[0] = Constant(0.0)
        
        return gg
    

    
    def nullspace(self):
      
        # No null space
        #null = False
        null = True
        return null


def single_run(paramsfile, mesh):

    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    #solver parameters:
    
    dt = 1.0/40
    T = 1.0
    
    #problem parameters
    dictionary = {}
    with open(paramsfile) as f:
        for line in f:
            print type(line.split()[0])
            print type(line.split()[1])
            
            dictionary.update({line.split()[0]:eval(line.split()[1])})

    print mesh
    dictionary.update({"mesh_file":str(mesh)})
    print dictionary
    # print dictionary["M"]
 
    #Loading mesh from file

    # filename = "../../meshes/2D_brain_refined6.xdmf"
    # f = XDMFFile(mpi_comm_world(), filename)
    # f.read(mesh, True)
    # mesh = refine(mesh)
    # Create problem set-up 
    # problem = FirstTest(dict(mesh_file=mesh, L=L, Q=Q, AA=AA, alphas=alphas,
    #                          Ks=Ks, G=G, Acceleration=Acceleration, Incompressible=Incompressible, E=E, nu=nu))

    problem = FirstTest(dictionary)
    
    # problem.mesh = refine(problem.mesh)
    print "lmbda = ", problem.params.nu*problem.params.E/((1.0-2.0*problem.params.nu)*(1.0+problem.params.nu))
    print "mu = ", problem.params.E/(2.0*(1.0+problem.params.nu))

    print "Ks = ", problem.params.Ks

    foldername = "BrainMesh_PulsatilePressure_FixedBrain_Acceleration_%s"%problem.params.Acceleration + "_nullspace_%s" %problem.nullspace() +\
                 "_E_" + "%05.03e" %problem.params.E + "_nu_" + "%04.05e" %problem.params.nu +"_K_" + "%04.05e" %problem.params.Ks[0] +\
                 "_Q" + "%04.03e" %problem.params.Q + "_Donut_coarse_refined"
    
    

    Vf = VectorFunctionSpace(problem.mesh, "CG", 1)
    vfinterp = Function(Vf)
    
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 1.0, "direct_solver":False, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})


    # Solve
    #filenameu = "results/" + foldername + "/u_2D.xdmf"
    #filenamep0 = "results/" + foldername + "/p0_2D.xdmf"
    #filenamevf = "results/" + foldername + "/vf_2D.xdmf"
    
    #fileu = XDMFFile(mpi_comm_world(), filenameu)
    #filep0 = XDMFFile(mpi_comm_world(), filenamep0)
    #filevf = XDMFFile(mpi_comm_world(), filenamevf)

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
        p = U.split(deepcopy=True)[1:problem.params.AA+1]
        vf = -problem.params.Ks[0] * grad(p[0])
        vf = project(vf, Vf)

        fileu << u
        filep0 << p[0]
        filevf << vf
        
        #fileu.write(u.vector[:])
        #filep0.write(p[0])
        #filevf.write(vf)
        # print u(Point(30.0,0.0))
        # plot(u, key="u", title="displacement", mesh=mesh)
        
    problemparamsfile = "results/" + foldername + "/problemparams.txt"
    with open(problemparamsfile, "a") as f_problem: 
        for i in problem.params.keys():            
            f_problem.write(i + " " + str(problem.params[i]) + "\n")
    f_problem.close()

if __name__ == "__main__":

    paramsfile=sys.argv[1]
    mesh = sys.argv[2]
    single_run(paramsfile, mesh)

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
