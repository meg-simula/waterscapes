__author__ = "Eleonora Piersanti <eleonora@simula.no>"

# Modified by Marie E. Rognes <meg@simula.no>, 2017

from dolfin import *

from mpet.rm_basis_L2 import rigid_motions

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

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
        c_a p_a_t + alpha_a div(u_t) - div K_a grad p_a + sum_{b} S_ab (p_b - p_a) = g_a   (2)

    where 
    
      sigma(u) = 2*mu*eps(u) + lmbda div(u) I 

    and eps(u) = sym(grad(u)), and mu and lmbda are the standard Lame
    parameters. For each network a, c_a is the saturation coefficient,
    alpha_a is the Biot-Willis coefficient, and K_a is the hydraulic
    conductivity.

    f is a given body force and g_a source(s) in network a.

    See e.g. Tully and Ventikos, 2011 for further details on the
    multiple-network poroelasticity equations.

    Boundary conditions:

    We assume that there is a facet function marking the different
    subdomains of the boundary. 

    For the momentum equation (1):
    
    We assume that each part of the boundary of the domain is one of
    the following types:

    Dirichlet (dO_m_D): 

      u(., t) = \bar u(t) 

    Neumann (dO_m_N):

      (sigma(u) - sum_{a} alpha_a p_a I) * n = s

    Assume that a FacetFunction indicates the different boundaries,
    and that only the Neumann boundary dO_m_N is marked by 1.

    For the continuity equations (2):

    Dirichet (dO_c_a_D):

      p_a(., t) = \bar p_a(t) 
      
    Neumann (dO_c_a_N)

      K grad p_a(., t) * n = I_a(t)

    Robin  (dO_c_a_R)
     ...

    Assume that for each a, a FacetFunction indicates the different
    boundaries, and that only the Neumann boundary dO_c_a_N is marked
    by 1.

    Initial conditions:

      u(x, t_0) = u_0(x)

      p_a(x, t_0) = p0_a(x) if c_a > 0

    Variational formulation (using Einstein summation notation over a
    in the elliptic equation below):

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

        # Initialize objects and store
        F, L0, L1, L2, up_, up = self.create_variational_forms()
        self.F = F
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.up_ = up_
        self.up = up

    def create_dirichlet_bcs(self):
        """Extract information about Dirichlet boundary conditions from given
        MPET problem.

        """
        
        VP = self.up.function_space()

        # Boundary conditions for momentum equation
        bcs0 = []
        markers = self.problem.momentum_boundary_markers
        u_bar = self.problem.u_bar
        bcs0 += [DirichletBC(VP.sub(0), u_bar, markers, 0)]

        # Boundary conditions for continuity equation
        bcs1 = []
        p_bar = self.problem.p_bar
        for i in range(self.problem.params["A"]):
            markers = self.problem.continuity_boundary_markers[i]
            bcs1 += [DirichletBC(VP.sub(i+1), p_bar[i], markers, 0)]

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
        params.add("stabilization", False)
        #params.add(KrylovSolver.default_parameters())
        #params.add(LUSolver.default_parameters())
        #params.add("testing", False)
        #params.add("fieldsplit", False)
        #params.add("symmetric", False)
        return params

    def create_variational_forms(self):

        # Extract mesh from problem
        mesh = self.problem.mesh

        # Extract time step
        dt = Constant(self.params.dt)
        
        # Extract the number of networks
        A = self.problem.params["A"]

        # Create function spaces 
        V = VectorElement("CG", mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", mesh.ufl_cell(), self.params.p_degree)

        u_nullspace = self.problem.displacement_nullspace
        p_nullspace = self.problem.pressure_nullspace
        dimQ = sum(p_nullspace)
        if u_nullspace:
            debug("Nullspace for u detected")
            Z = rigid_motions(self.problem.mesh)
            dimZ = len(Z)
            RU = VectorElement('R', mesh.ufl_cell(), 0, dimZ)
            if dimQ:
                debug("Nullspace for p detected")
                RP = [FiniteElement('R', mesh.ufl_cell(), 0)
                      for i in range(dimQ)]
                M = MixedElement([V] + [W for i in range(A)] + [RU] + RP)
            else:
                M = MixedElement([V] + [W for i in range(A)] + [RU])
        else:
            if dimQ:
                debug("Nullspace for p, but not for u detected")
                RP = [FiniteElement('R', mesh.ufl_cell(), 0)
                      for i in range(dimQ)]
                M = MixedElement([V] + [W for i in range(A)] + RP)
            else:
                debug("Constructing standard variational form")
                M = MixedElement([V] + [W for i in range(A)])

        debug("Constructing Function Space")
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
        theta = self.params.theta
        um = theta*u + (1.0 - theta)*u_
        pm = [(theta*p[i] + (1.0-theta)*p_[i]) for i in range(A)]
        
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
        # boundary flux I from problem description
        f = self.problem.f
        g = self.problem.g
        s = self.problem.s
        I = self.problem.I
        beta = self.problem.beta
        p_robin = self.problem.p_robin
        # Define variational form to be solved at each time-step.
        dx = Measure("dx", domain=mesh)

        debug("Assembling form")
        As = range(A)
        F = inner(sigma(u), sym(grad(v)))*dx() \
            + sum([-alpha[i]*p[i]*div(v) for i in As])*dx() \
            + sum([- c*(p[i] - p_[i])*w[i] for i in As])*dx() \
            + sum([-alpha[i]*div(u-u_)*w[i] for i in As])*dx() \
            + sum([-dt*K[i]*inner(grad(pm[i]), grad(w[i])) for i in As])*dx() \
            + sum([sum([-dt*S[i][j]*(pm[i] - pm[j])*w[i] for j in As]) \
                   for i in As])*dx() \

        stabilization = self.params.stabilization
        if stabilization == True:
            h = mesh.hmin()
            mu = E/(2.0*((1.0 + nu)))
            lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
            F += inner(sum([-h*h*K[i]/4.0(lmbda+2*mu)*grad(p[i]-p_[i])*w[i] for i in As]) )
            
        # Add orthogonality vefrsus rigid motions if nullspace for the
        # displacement
        debug("Assembling nullspace for u")
        if u_nullspace:
            F += sum(r[i]*inner(Z[i], u)*dx() for i in range(dimZ)) \
                 + sum(z[i]*inner(Z[i], v)*dx() for i in range(dimZ))

        # Add orthogonality versus constants if nullspace for the
        # displacement
        if dimQ:
            i = 0
            for (k, p_nullspace) in enumerate(self.problem.pressure_nullspace):
                if p_nullspace:
                    F += p[k]*w_null[i]*dx() + p_null[i]*w[k]*dx()
                    i += 1
        
        # Add body force and traction boundary condition for momentum equation
        NEUMANN_MARKER = 1
        ROBIN_MARKER = 2
        markers = self.problem.momentum_boundary_markers
        dsm = Measure("ds", domain=mesh, subdomain_data=markers)
        L0 = dot(f, v)*dx() + inner(s, v)*dsm(NEUMANN_MARKER)
        # Add source and flux boundary conditions for continuity equations
        dsc = []
        L1 = []
        L2 = []
        info("Assembling Neumann and Robin rhs")
        for i in As:
            markers = self.problem.continuity_boundary_markers[i]
            dsc += [Measure("ds", domain=mesh, subdomain_data=markers)]
            L1 += [dt*g[i]*w[i]*dx() + dt*I[i]*w[i]*dsc[i](NEUMANN_MARKER)]
                  
            L2 += [dt*beta[i]*(-pm[i]+p_robin[i])*w[i]*dsc[i](ROBIN_MARKER)]
        # Set solution field(s)
        up = Function(VW)
        
        return F, L0, L1, L2, up_, up

    def solve(self):
        """Solve given MPET problem, yield solutions at each time step.

        Assumptions:
        - Users should set self.up_ to be the initial conditions for up;
        """
        
        dt = self.params["dt"]
        T = self.params["T"]
        theta = self.params["theta"]
        time = self.problem.time

        # Extract lhs a and implicitly time-dependent rhs L
        (a, L) = system(self.F)
        L0 = self.L0
        L1 = self.L1
        L2 = self.L2
        # Extract essential bcs
        [bcs0, bcs1] = self.create_dirichlet_bcs()
        bcs = bcs0 + bcs1
        
        # Assemble left-hand side matrix
        
        info("Assembling a")        
        A = assemble(a) 

        info("Assembling L2")        

        for L2i in L2: 
                A2 = assemble(lhs(L2i))
                A.axpy(1.0, A2, False)
        
        # Create solver
        solver = LUSolver(A)
        
        # Start with up as up_, can help Krylov Solvers
        self.up.assign(self.up_)

        while (float(time) < (T - 1.e-10)):

            # Handle the different parts of the rhs a bit differently
            # due to theta-scheme
            b = assemble(L)

            # Set t_theta to t + dt (when theta = 1.0) or t + 1/2 dt
            # (when theta = 0.5)
            t_theta = float(time) + theta*float(dt)
            time.assign(t_theta)                

            # Assemble time-dependent rhs for parabolic equations
            for L1i in L1: 
                b1 = assemble(L1i)
                b.axpy(1.0, b1)
            
            for L2i in L2: 
                b2 = assemble(rhs(L2i))
                b.axpy(1.0, b2)    
            # Set t to "t"
            t = float(time) + (1.0 - theta)*float(dt)
            time.assign(t)
            
            # Assemble time-dependent rhs for elliptic equations
            b0 = assemble(L0)    
            b.axpy(1.0, b0)

            # Apply boundary conditions
            for bc in bcs:
                bc.apply(A, b)
            
            # Solve
            solver.solve(self.up.vector(), b)

            # Yield solution and time
            yield self.up, float(time)

            # Update previous solution up_ with current solution up
            self.up_.assign(self.up)

            # Update time
            time.assign(t)
        
