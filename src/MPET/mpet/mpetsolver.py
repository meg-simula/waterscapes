__author__ = "Eleonora Piersanti <eleonora@simula.no>"

# Modified by Marie E. Rognes <meg@simula.no>, 2017

from numpy import random

from dolfin import *

from mpet.rm_basis_L2 import rigid_motions
from mpet.bc_symmetric import *

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Marker conventions
DIRICHLET_MARKER = 0
NEUMANN_MARKER = 1
ROBIN_MARKER = 2

def elastic_stress(u, E, nu):
    "Define the standard linear elastic constitutive equation."
    d = u.geometric_dimension()
    I = Identity(d)
    mu = E/(2.0*((1.0 + nu)))
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    s = 2*mu*sym(grad(u)) + lmbda*div(u)*I
    return s

class MPETSolver(object):
    """This solver solves the multiple-network poroelasticity equations
    (MPET): find a vector field (the displacement) u and the network
    pressures p_a for a set of networks a = 1, ..., A such that:

        - div ( sigma(u) - sum_{a} alpha_a p_a I) = f           (1)
        c_a p_a_t + alpha_a div(u_t) - div K_a grad p_a + sum_{b} S_ab (p_a - p_b) = g_a   (2)

    where 
    
      sigma(u) = 2*mu*eps(u) + lmbda div(u) I 

    and eps(u) = sym(grad(u)), and mu and lmbda are the standard Lame
    parameters. For each network a, c_a is the specific storage
    coefficient, alpha_a is the Biot-Willis coefficient, and K_a is
    the hydraulic conductivity.

    f is a given body force and g_a source(s) in network a.

    See e.g. Tully and Ventikos, 2011 for further details on the
    multiple-network poroelasticity equations.

    Boundary conditions:

    We assume that there are (possibly multiple) facet functions
    marking the different subdomains of the boundary. We assume that
    Dirichlet conditions are marked by 0, Neumann conditions marked by
    1 and Robin conditions marked by 2.

    For the momentum equation (1):
    
    We assume that each part of the boundary of the domain is one of
    the following two types:

    *Dirichlet*: 

      u(., t) = \bar u(t) 

    *Neumann*:

      (sigma(u) - sum_{a} alpha_a p_a I) * n = s

    Assume that for each a, a FacetFunction indicates the different
    boundaries, and that the Dirichlet boundary is marked by 0, the
    Neumann boundary is marked by 1 and the Robin boundary is marked
    by 2.

    For the continuity equations (2):

    We assume that each part of the boundary of the domain is one of
    the following three types:

    *Dirichet*

      p_a(., t) = \bar p_a(t) 
      
    *Neumann*

      K grad p_a(., t) * n = I_a(t)

    *Robin* 
       
      K grad p_a(., t) * n = \beta_a (p_a - p_a_ robin)
    
    Assume that for each a, a FacetFunction indicates the different
    boundaries, and that the Dirichlet boundary is marked by 0, the
    Neumann boundary is marked by 1 and the Robin boundary is marked
    by 2.

    Initial conditions:

      u(x, t_0) = u_0(x)

      p_a(x, t_0) = p0_a(x) if c_a > 0

    Variational formulation (using Einstein summation notation over a
    in the elliptic equation below):

       S MISSING IN FORMULATION BELOW.
       ROBIN MISSING IN FORMULATION BELOW.

    Find u(t) and p_a(t) such that

      <sigma(u), eps(v)> - <alpha_a p_a, div v> - < (sigma(u) - alpha_a p_a I) * n, v>_dO = <f, v>   for all v in V 

      <c_a p_a_t + alpha_a div(u_t), q_a > + <K_a grad p_a, grad q_a> - <K_a grad p_a * n, q_a>_dO = <g_a, q_a> for all q_a in Q_a, all a

    Inserting boundary conditions: 

    Find u(t) in V such that u(t) = \bar u(t) on dO_m_D and p_a(t) in Q_a such that p_a = \bar p_a on dO_c_a_D such that 

                            <sigma(u), eps(v)> - <alpha_a p_a, div v> = <f, v> + <s, v>_dO_m_N  for all v  in V such that v = 0 on dO_m_D

      <c_a p_a_t + alpha_a div(u_t), q_a > + <K_a grad p_a, grad q_a> = <g_a, q_a> + <I_a, q_a>_dO_c_a_N for all q_a in Q_a, all a

    """

    def __init__(self, problem, params=None):
        "Create solver with given MPET problem and parameters."
        self.problem = problem
        # Update parameters if given
        self.params = self.default_params()
        if params is not None:
            self.params.update(params)

        self.solver_monitor = {}
        # Initialize variational forms and store
        a, a_robin, L, L0, L1, prec, up_, up = self.create_variational_forms()

        self.a = a             # Main left-hand side form
        self.a_robin = a_robin # List of left-hand side forms for Robin boundary conditions
        self.L = L             # Right-hand side form that does not depend explicitly on time
        self.L0 = L0           # List of right-hand side forms (from elliptic momentum equation)
        self.L1 = L1           # List of right-hand side forms (from parabolic continuity equation)

        self.up_ = up_         # Solution at previous time step
        self.up = up           # Solution at current time step

        self.prec = prec

    def create_dirichlet_bcs(self):
        """Extract information about Dirichlet boundary conditions from given
        MPET problem.

        """
        VP = self.up.function_space()

        # Define and create boundary conditions for momentum equation
        bcs0 = []
        markers = self.problem.momentum_boundary_markers
        u_bar = self.problem.u_bar
        bcs0 += [DirichletBC(VP.sub(0), u_bar, markers, DIRICHLET_MARKER)]

        # Define and create Boundary conditions for the continuity equations
        bcs1 = []
        p_bar = self.problem.p_bar
        for i in range(self.problem.params["A"]):
            markers = self.problem.continuity_boundary_markers[i]
            bcs1 += [DirichletBC(VP.sub(i+1), p_bar[i], markers, DIRICHLET_MARKER)]

        return [bcs0, bcs1]
    
    @staticmethod
    def default_params():
        "Define default solver parameters."
        params = Parameters("MPETSolver")
        params.add("dt", 0.05)
        params.add("t", 0.0)
        params.add("T", 1.0)
        params.add("theta", 0.5)
        params.add("u_degree", 2)
        params.add("p_degree", 1)
        params.add("direct_solver", True)
        params.add("testing", False)
        return params

    def create_variational_forms(self):
        "."
        
        # Extract mesh from problem
        mesh = self.problem.mesh

        # Extract time step
        dt = Constant(self.params["dt"])
        
        # Extract the number of networks
        A = self.problem.params["A"]
        As = range(A)

        # Create function spaces 
        V = VectorElement("CG", mesh.ufl_cell(), self.params["u_degree"])
        W = FiniteElement("CG", mesh.ufl_cell(), self.params["p_degree"])
        
        u_nullspace = self.problem.displacement_nullspace
        p_nullspace = self.problem.pressure_nullspace
        dimQ = sum(p_nullspace)
        if u_nullspace:
            Z = rigid_motions(self.problem.mesh)
            dimZ = len(Z)
            RU = VectorElement('R', mesh.ufl_cell(), 0, dimZ)
            if dimQ:
                RP = [FiniteElement('R', mesh.ufl_cell(), 0)
                      for i in range(dimQ)]
                M = MixedElement([V] + [W for i in As] + [RU] + RP)
            else:
                M = MixedElement([V] + [W for i in As] + [RU])
        else:
            if dimQ:
                RP = [FiniteElement('R', mesh.ufl_cell(), 0)
                      for i in range(dimQ)]
                M = MixedElement([V] + [W for i in As] + RP)
            else:
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
        
        # Extract test and trial functions corresponding to the
        # nullspace Lagrange multiplier
        if u_nullspace == True:
            z = up[-1-dimQ]
            r = vw[-1-dimQ]
            if dimQ:
                p_null = up[-1-dimQ+1:]
                w_null = vw[-1-dimQ+1:]
        else:
            if dimQ:
                p_null = up[-1-dimQ+1:]
                w_null = vw[-1-dimQ+1:]
            else:
                pass
                
        # um and pm represent the solutions at time t + dt*theta
        theta = self.params["theta"]
        um = theta*u + (1.0 - theta)*u_
        pm = [(theta*p[i] + (1.0-theta)*p_[i]) for i in As]
        
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
            + sum([-dt*K[i]*inner(grad(pm[i]), grad(w[i])) for i in As])*dx() \
            + sum([sum([-dt*S[i][j]*(pm[i] - pm[j])*w[i] for j in As]) \
                   for i in As])*dx() 

        # Define form for preconditioner
        prec = 0
        if not self.params["direct_solver"]:
            info("Defining preconditioner")
            mu = E/(2.0*((1.0 + nu)))
            pu = mu * inner(grad(u), grad(v))*dx() 
            pp = sum([c[i]*p[i]*w[i]*dx() + dt*theta* K[i]*inner(grad(p[i]), grad(w[i]))*dx() \
                      + sum([dt*theta*S[i][j] for j in list(As[:i])+list(As[i+1:])])*p[i]*w[i]*dx() for i in As])
            prec += pu + pp

        # Add orthogonality versus rigid motions if nullspace for the
        # displacement
        if u_nullspace:
            F += sum(r[i]*inner(Z[i], u)*dx() for i in range(dimZ)) \
                 + sum(z[i]*inner(Z[i], v)*dx() for i in range(dimZ))
            if not self.params["direct_solver"]:     
                prec += sum(z[i]*r[i]*dx() for i in range(dimZ)) + inner(u,v)*dx() 
            
        # Add orthogonality versus constants if nullspace for the
        # displacement
        if dimQ:
            i = 0
            for (k, p_nullspace) in enumerate(self.problem.pressure_nullspace):
                if p_nullspace:
                    F += p[k]*w_null[i]*dx() + p_null[i]*w[k]*dx()
                    if not self.params["direct_solver"]:
                        prec += p_null[i]*w_null[i]*dx() + p[k]*w[k]*dx() 
                    i += 1
                    
        # Add body force and traction boundary condition for momentum
        # equation. The form L0 holds the right-hand side terms of the
        # momentum (elliptic) equation, which may depend on time
        # explicitly and should be evaluated at time t + dt
        markers = self.problem.momentum_boundary_markers
        dsm = Measure("ds", domain=mesh, subdomain_data=markers)
        L0 = dot(f, v)*dx() + inner(s, v)*dsm(NEUMANN_MARKER)

        # Define forms including sources and flux boundary conditions
        # for continuity equations. The list of forms L1 holds the
        # right-hand side terms of the continuity (parabolic)
        # equations, which may depend on time explicitly, and should
        # be evaluated at t + theta*dt
        dsc = []
        L1 = []
        a_robin = []
        info("Defining contributions from Neumann and Robin boundary conditions")
        for a in As:
            markers = self.problem.continuity_boundary_markers[a]
            dsc += [Measure("ds", domain=mesh, subdomain_data=markers)]

            # Add Neumann contribution to list L1
            L1 += [dt*g[a]*w[a]*dx() + dt*I[a]*w[a]*dsc[a](NEUMANN_MARKER)]

            # Add Robin contributions to both F and to L1 
            F2a = dt*beta[a]*(-pm[a] + p_robin[a])*w[a]*dsc[a](ROBIN_MARKER)
            a_robin += [lhs(F2a)]
            L1 += [rhs(F2a)]

        # Define function for current solution
        up = Function(VW)

        # Just split main form F here into a and L
        a = lhs(F)
        L = rhs(F)

        return a, a_robin, L, L0, L1, prec, up_, up

    def solve(self):
        """Solve the given MPET problem to the end time given by the parameter
        'T'. This method yields solutions at each time step.

        Users must set 'up_' to the correct initial conditions prior
        to calling solve.

        """
        if self.params["direct_solver"]:
            return self.solve_direct()
        else:
            return self.solve_iterative()

    def solve_direct(self):
        """Solve the given MPET problem to the end time given by the parameter
        'T' using a direct (LU) solver. This method yields solutions
        at each time step.

        Users must set 'up_' to the correct initial conditions prior
        to calling solve.

        """

        # Extract parameters related to the time-stepping
        dt = self.params["dt"]
        T = self.params["T"]
        theta = self.params["theta"]
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
        solver.parameters['reuse_factorization'] = True 
        
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

    def solve_iterative(self):
        """Solve the given MPET problem to the end time given by the parameter
        'T' using a preconditioned Krylov solver. This method yields
        solutions at each time step.

        Users must set 'up_' to the correct initial conditions prior
        to calling solve.

        """

        # Extract parameters related to the time-stepping
        dt = self.params["dt"]
        T = self.params["T"]
        theta = self.params["theta"]
        time = self.problem.time

        # Extract relevant variational forms
        a = self.a
        a_robin = self.a_robin
        L = self.L
        L0 = self.L0
        L1 = self.L1
        prec = self.prec

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

        # Assemble preconditioner and apply boundary conditions
        P = assemble(prec)
        for bc in bcs:
            apply_symmetric(bc, P)

        # Create KrylovSolver
        solver = PETScKrylovSolver("minres", "hypre_amg")
        self.solver_monitor["niter"] = []
        # Assign initial conditions (in up_) to current solution as a
        # good starting guess for iterative solver
        if self.params["testing"]:
            self.up.vector()[:] = random.randn(self.up.vector().size())     
        else:
            self.up.assign(self.up_)

        while (float(time) < (T - 1.e-9)):

            # Copy the non-boundary conditions applied matrix A
            Acopy = A.copy()

            # Times defining the time interval (for readability)
            t0 = float(time)
            t_theta = t0 + theta*float(dt)
            t1 = t0 + float(dt)
            
            # 1. Assemble the parts of right-hand side forms (i.e. L)
            # that does not depend on time explicitly
            b = assemble(L)  

            # Set t_theta to t + dt (when theta = 1.0) or t + 1/2 dt
            # (when theta = 0.5)
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

            # Apply boundary conditions symmetrically            
            for bc in bcs:
                bc.apply(b)    
                apply_symmetric(bc, Acopy, b)

            # Give updated matrix and preconditioner P to solver
            solver.set_operators(Acopy, P)

            # Solve
            niter = solver.solve(self.up.vector(), b)
            self.solver_monitor["niter"] += [niter]
            
            # Yield solution and time
            yield self.up, float(time)

            # Update previous solution up_ with current solution up
            self.up_.assign(self.up)

            # Update time
            time.assign(t1)

        self.solver_monitor["P"] = P
        self.solver_monitor["A"] = Acopy
    
