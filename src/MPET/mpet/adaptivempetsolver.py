__author__ = "Marie E. Rognes <meg@simula.no>"

from numpy import random

from dolfin import *

from mpetsolver import MPETSolver, elastic_stress, DIRICHLET_MARKER, NEUMANN_MARKER
from mpetproblem import MPETProblem

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class AdaptiveMPETSolver(MPETSolver):
    """

    """

    def __init__(self, problem, params=None):
        "Create solver with given MPET problem and parameters."
        self.problem = problem
        # Update parameters if given
        self.params = self.default_params()
        if params is not None:
            self.params.update(params)

        (a, L, up) = self.create_variational_forms()
            
    @staticmethod
    def default_params():
        "Define default solver parameters."
        params = Parameters("AdaptiveMPETSolver")
        params.add("dt", 0.05)
        params.add("t", 0.0)
        params.add("T", 1.0)
        return params

    def create_variational_forms(self):
        "."
        
        # Extract mesh from problem
        mesh = self.problem.mesh

        # Extract time step (will be variable)
        dt = Constant(self.params["dt"])
        
        # Extract the number of networks
        A = self.problem.params["A"]
        As = range(A)

        # Create function spaces (ignoring null spaces for now)
        V = VectorElement("CG", mesh.ufl_cell(), 2)
        W = FiniteElement("CG", mesh.ufl_cell(), 1)
        M = MixedElement([V] + [W for i in As])
        VW = FunctionSpace(mesh, M)

        # Create previous solution field(s) and extract previous
        # displacement solution u_ and pressures p_ = (p_1, ..., p_A)
        up_ = Function(VW)
        u_ = split(up_)[0]
        p_ = split(up_)[1:A+1]
        
        # Create trial functions and extract displacement u and pressure
        # trial functions p = (p_1, ..., p_A)
        up = TrialFunctions(VW)
        u = up[0]
        p = up[1:A+1]

        # Create test functions and extract displacement u and pressure
        # test functions p = (p_1, ..., p_A)
        vw = TestFunctions(VW)
        v = vw[0]
        w = vw[1:A+1]
        
        # Extract material parameters from problem
        E = self.problem.params["E"]          
        nu = self.problem.params["nu"]        
        alpha = self.problem.params["alpha"]  
        K = self.problem.params["K"]
        S = self.problem.params["S"]
        c = self.problem.params["c"]

        # Define the extra/elastic stress
        sigma = lambda u: elastic_stress(u, E, nu)

        # Extract body force f and sources g, boundary traction s and
        # boundary flux I, boundary Robin coefficient beta(s) and
        # Robin pressures p_robin from problem description
        f = self.problem.f
        g = self.problem.g
        s = self.problem.s
        I = self.problem.I
        beta = self.problem.beta
        p_robin = self.problem.p_robin

        # Define main variational form to be solved at each time-step.
        dx = Measure("dx", domain=mesh)
        F = inner(sigma(u), sym(grad(v)))*dx() \
            + sum([-alpha[i]*p[i]*div(v) for i in As])*dx() \
            + sum([-c[i]*(p[i] - p_[i])*w[i] for i in As])*dx() \
            + sum([-alpha[i]*div(u-u_)*w[i] for i in As])*dx() \
            + sum([-dt*inner(K[i]*grad(p[i]), grad(w[i])) for i in As])*dx() \
            + sum([sum([-dt*S[i][j]*(p[i] - p[j])*w[i] for j in As]) \
                   for i in As])*dx() 
        
        # Add body force and traction boundary condition for momentum
        # equation. The form L0 holds the right-hand side terms of the
        # momentum (elliptic) equation, which may depend on time
        # explicitly and should be evaluated at time t + dt
        markers = self.problem.momentum_boundary_markers
        dsm = Measure("ds", domain=mesh, subdomain_data=markers)
        F -= dot(f, v)*dx() + inner(s, v)*dsm(NEUMANN_MARKER)

        # Define forms including sources and flux boundary conditions
        # for continuity equations.
        dsc = []
        info("Defining contributions from Neumann boundary conditions")
        for a in As:
            markers = self.problem.continuity_boundary_markers[a]
            dsc += [Measure("ds", domain=mesh, subdomain_data=markers)]

            # Subtract Neumann contribution from F
            F -= dt*g[a]*w[a]*dx() + dt*I[a]*w[a]*dsc[a](NEUMANN_MARKER)

        # Define function for current solution
        up = Function(VW)

        # Just split main form F here into a and L
        a = lhs(F)
        L = rhs(F)

        return a, L, up

    def step(self, dt):
        """Solve the given MPET problem from the current time with time step
        'dt'. Users must set 'up_' to the correct initial conditions
        prior to calling 'step'.

        """

        # Extract parameters related to the time-stepping
        time = self.problem.time

        # Extract relevant variational forms
        a = self.a
        a_robin = self.a_robin
        L = self.L
        L0 = self.L0
        L1 = self.L1

        # Create essential bcs
        [bcs0, bcs1] = self.create_dirichlet_bcs()
        bcs = bcs0 + bcs1
        
        # Assemble left-hand side matrix including Robin
        # terms. (Design due to missing FEniCS feature of assembling
        # integrals of same type with different subdomains.)
        A = assemble(a)  
        for a_a in a_robin:
            A_a = assemble(a_a)
            A.axpy(1.0, A_a, False) 
        
        # Apply boundary conditions to matrix once:
        for bc in bcs:
            bc.apply(A)
        
        # Create LU solver and reuse LU factorization
        solver = LUSolver(A, "mumps")
        
        while (float(time) < (T - 1.e-9)):

            # Times defining the time interval (for readability)
            t0 = float(time)
            t_theta = t0 + theta*float(dt)
            t1 = t0 + float(dt)
            
            # 1. Assemble the parts of right-hand side forms (i.e. L)
            # that does not depend on time explicitly
            b = assemble(L)  

            # Update time to t0 + theta*dt
            time.assign(t_theta)                

            # 2. Assemble time-dependent rhs for parabolic equations
            # and add to right-hand side vector b
            for l in L1: 
                b_l = assemble(l)  
                b.axpy(1.0, b_l)

            # Update time to t1:
            time.assign(t1)
            
            # 3. Assemble time-dependent rhs for elliptic equations
            b0 = assemble(L0)
            b.axpy(1.0, b0)

            # Apply boundary conditions            
            for bc in bcs:
                bc.apply(b)

            # Solve
            solver.solve(A, self.up.vector(), b)

            # Yield solution and time
            yield self.up, float(time)

            # Update previous solution up_ with current solution up
            self.up_.assign(self.up)

            # Update time
            time.assign(t1)



if __name__ == "__main__":

    mesh = UnitSquareMesh(2, 2)
    time = Constant(0.0)
    E, nu = 3.0, 0.45
    material = dict(E=E, nu=nu, alpha=(0.5, 0.5), A=2, c=(1.0, 1.0), K=(1.0, 1.),
                    S=((0.0, 1.0), (1.0, 0.0)))
    problem = MPETProblem(mesh, time, params=material)
    solver = AdaptiveMPETSolver(problem)
