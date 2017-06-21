#from time import time

from dolfinimport import *
import numpy as np
#from block import *
#from block.algebraic.petsc import *
#from block.iterative import *
from rm_basis_L2 import rigid_motions 
import resource

class MPETSolver(object):
    """This solver solves the multiple-network poroelasticity equations
    (MPET): find a vector field (the displacement) u and the network
    pressures p_a for a set of networks a = 1, ..., A such that:

        - div ( sigma(u) - sum_{a} alpha_a p_a ) = f     (1)
        c_a p_t + alpha_a div(u_t) - div K_a p_a = g_a   (2)

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

    Dirichlet: 

      u(., t) = \bar u(t) 

    Neumann:

      (sigma(u) - sum_{a} alpha_a p_a I) * n = traction

    For the continuity equations (2):

    Dirichet:

      p_a(., t) = \bar p_a(t) 
      
    Neumann

      K grad p_a(., t) * n = flux_a(t)

    Robin

     ...

    Initial conditions:

      u(x, t_0) = u_0(x)

      p_a(x, t_0) = p0_a(x) if c_a > 0
    """

    def __init__(self, problem, params=None):
        "Create solver with given MPET problem and parameters."
        self.problem = problem

        # Redefine measures based on subdomain information provided by
        # the problem
        self.ds = Measure("ds", domain=self.problem.mesh,
                          subdomain_data=self.problem.facet_domains)
        self.dx = Measure('dx', domain=self.problem.mesh)

        # Update parameters if given
        self.params = self.default_params()
        if params is not None:
            self.params.update(params)
                
    @staticmethod
    def default_params():
        "Define default solver parameters."
        params = Parameters("SimpleSolver")
        params.add("dt", 0.05)
        params.add("t", 0.0)
        params.add("T", 1.0)
        params.add("theta", 0.5)
        params.add("u_degree", 2)
        params.add("p_degree", 1)
        params.add("direct_solver", True)
        params.add(KrylovSolver.default_parameters())
        params.add(LUSolver.default_parameters())
        params.add("testing", False)
        params.add("fieldsplit", False)
        params.add("symmetric", False)
        return params

    def initialize_forms(self):

        # Extract the number of networks
        A = self.problem.params.AA

        # Create function spaces
        V = VectorElement("CG", self.problem.mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", self.problem.mesh.ufl_cell(), self.params.p_degree)

        nullspace = self.problem.nullspace()
        if nullspace == True:
            Z = rigid_motions(self.problem.mesh)
            R = VectorElement('R', self.problem.mesh.ufl_cell(), 0, len(Z))
            MX = MixedElement([V] + [W for i in range(AA)] + [R])
            ME = FunctionSpace(self.problem.mesh, MX)

        else:
            MX = MixedElement([V] + [W for i in range(AA)])
            ME = FunctionSpace(self.problem.mesh, MX)

    
    def solve(self):

        # Extract number of networks (AA)
        AA = self.problem.params.AA
        # Create function spaces
        V = VectorElement("CG", self.problem.mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", self.problem.mesh.ufl_cell(), self.params.p_degree)

        V0 = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W0 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        Ve = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree+2)
        We = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree+2)

        nullspace = self.problem.nullspace()
        if nullspace == True:
            Z = rigid_motions(self.problem.mesh)
            R = VectorElement('R', self.problem.mesh.ufl_cell(), 0, len(Z))
            MX = MixedElement([V] + [W for i in range(AA)] + [R])
            ME = FunctionSpace(self.problem.mesh, MX)

        else:
            MX = MixedElement([V] + [W for i in range(AA)])
            ME = FunctionSpace(self.problem.mesh, MX)

        # Create solution field(s)
        U0 = Function(ME, name="U0") # Solution at t = n-1
        u0 = split(U0)[0]
        ps0 = split(U0)[1:AA+1]

        U_1 = Function(ME, name="U_1") # Solution at t = n
        u_1= split(U_1)[0]
        ps_1 = split(U_1)[1:AA+1]
        
        U = Function(ME, name="U") # Solution at t = n+1

        # Create trial functions
        solutions = TrialFunctions(ME)
        u = solutions[0]
        ps = solutions[1:AA+1]

        # Create test functions
        tests = TestFunctions(ME)
        v = tests[0]
        # q0, q1 = tests[1:]
        qs = tests[1:AA+1]

        if nullspace == True:
            zs = solutions[-1]
            rs = tests[-1]
            print "len(zs) = ", len(zs)
            
        # --- Define forms, material parameters, boundary conditions, etc. ---
        n = FacetNormal(self.problem.mesh)

        # Extract material parameters from problem object
        E = self.problem.params.E           # Young's modulus
        nu = self.problem.params.nu         # Poisson ratio
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
        alphas = self.problem.params.alphas # Biot coefficients (one per network)
        Q = self.problem.params.Q           # Compressibility coefficients (1)
        G = self.problem.params.G           # Transfer coefficents (scalar values, representing transfer from network i to j)
        Ks = self.problem.params.Ks         # Permeability tensors (one per network)

        # Extract discretization parameters
        theta = self.params.theta           # Theta-scheme parameters (0.5 CN, 1.0 Backward Euler)
        t = self.params.t
        dt = self.params.dt                 # Timestep (fixed for now)
        T = self.params.T                   # Final time

        time = Constant(t)                  # Current time
        time_dt = Constant(t+dt)            # Next time

        # um and pms represent the solutions at time t + dt*theta
        um = theta * u + (1.0 - theta) * u0
        pms = [ (theta * ps[i] + (1.0 - theta) * ps0[i]) for i in range(AA)]

        # Expression for the elastic stress tensor in terms of Young's
        # modulus E and Poisson's ratio nu
        def sigma(u):
            d = u.geometric_dimension()
            I = Identity(d)
            s = E/(1.0 + nu)*sym(grad(u)) + nu*E/((1.0-2.0*nu)*(1.0+nu))*div(u)*I
            return s

        # Define 'elliptic' right hand side at t + dt
        ff = self.problem.f(time_dt)

        # Define 'parabolic' right hand side(s) at t + theta*dt
        gg0 = self.problem.g(time)
        gg1 = self.problem.g(time_dt)
        gg = [(theta*gg1[i] + (1.0-theta)*gg0[i]) for i in range(AA)]

        # Extract the given initial conditions and interpolate into
        # solution finite element spaces
        ## MER: I find the 'time' argument a little funny, but keep it
        ## if you would like!
        u_init, ps_init = self.problem.initial_conditions(time)
        # u_init = interpolate(u_init, V)
        u_init = project(u_init, V0)
        # ps_init = [interpolate(ps_init_i, W) for ps_init_i in ps_init]
        ps_init = [project(ps_init_i, W0) for ps_init_i in ps_init]
        # Update the 'previous' solution with the initial conditions
        assign(U0.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U0.sub(i+1), ps_init_i)

        assign(U_1.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U_1.sub(i+1), ps_init_i)

        # --- Define the variational forms ---

        #Acceleration terms
               
        # Elliptic part
        a = inner(sigma(u), sym(grad(v)))*self.dx
        b = sum([- alphas[i]*ps[i]*div(v) for i in range(AA)])*self.dx
        f = dot(ff, v)*self.dx

        # Parabolic part(s)
        d =  sum([- alphas[i]*div(u - u0)*qs[i]
                 for i in range(AA)])*self.dx
        e = sum([- dt * Ks[i]*inner(grad(pms[i]), grad(qs[i]))
                 for i in range(AA)])*self.dx
        s = sum([sum([- dt * G[j]*(pms[i] - pms[j])*qs[i] for j in range(AA)])
                 for i in range(AA)]) * self.dx
        g =  sum([dt * gg[i]*qs[i] for i in range(AA)]) * self.dx

        # Combined variational form (F = 0 to be solved)
        
        F = a + b - f + d + e + s - g

        if self.problem.params.Acceleration == True:
            ddu_ddt = (u - 2*u0 + u_1)/(dt*dt)
            F += inner(ddu_ddt, v)*dx - sum([dt * div(Ks[i]*ddu_ddt)*qs[i] for i in range(AA)]) * self.dx
            
        ## If Q is infinity, the term below drops and we do not
        ## include it in the system. Oppositely, we add this term.
        if (self.problem.params.Incompressible == False):
            c = sum([-1.0/(Q)*(ps[i] - ps0[i])*qs[i] for i in range(AA)])*self.dx
            F += c

        if nullspace == True:
            F += sum(rs[i]*inner(Z[i], u)*self.dx for i in range(len(Z)))
            F += sum(zs[i]*inner(Z[i], v)*self.dx for i in range(len(Z)))

        # Create Dirichlet boundary conditions at t + dt Assuming that
        # boundary conditions are specified via a subdomain for now
        # (FIXME more general later)


        boundary_conditions_u = self.problem.boundary_conditions_u(time, time_dt, theta)
        boundary_conditions_p = self.problem.boundary_conditions_p(time, time_dt, theta)

        bcs = []
        integrals_N = []
        integrals_R = []

        for i in boundary_conditions_u:
            if 'Dirichlet' in boundary_conditions_u[i]:
                bc = DirichletBC(ME.sub(0), boundary_conditions_u[i]['Dirichlet'], self.problem.facet_domains, i)
                bcs.append(bc)
            if 'Neumann' in boundary_conditions_u[i]:
                stress = boundary_conditions_u[i]['Neumann']
                integrals_N.append(inner(stress*n,v)*self.ds(i))
            if 'Robin' in boundary_conditions_u[i]:
                r, ur = boundary_conditions[i]['Robin']
                integrals_R.append(r*inner((u - ur),v)*self.ds(i))

        for i in range(len(boundary_conditions_p)):
            bcsp = boundary_conditions_p[i]
            for j in bcsp:
                if 'Dirichlet' in bcsp[j]:
                    print "j = ", j
                    print "time = ", bcsp[j]
                    bc = DirichletBC(ME.sub(i+1), bcsp[j]["Dirichlet"], self.problem.facet_domains, j)
                    bcs.append(bc)
                if 'Neumann' in bcsp[j]:
                    flux = bcsp[j]['Neumann']
                    integrals_N.append(inner(dt * flux,n) * qs[i] * self.ds(j))
                if 'Robin' in bcsp[j]:
                    (r, pr) = bcsp[j]['Robin']
                    integrals_R.append(dt * r * (pms[i] - pr) * qs[i] * self.ds(j))

        F +=  sum(integrals_N) + sum(integrals_R)

        # FIXME: This needs testing
        #attention we must pass ds and dx of the problem!!!
        #neumann bc if there are neumann bc add them to the form
        #.....
        # ncsu, ncsp = self.problem.neumann_conditions(time, time_dt, theta)
        # if len(ncsu) > 0:
        #     for i in range(len(ncsu)):
        #         F += inner(ncsu[i][0]*n,v) * self.ds(ncsu[i][1])
        #         print assemble(1.0*self.ds(ncsu[i][1]))
        # if len(ncsp) > 0:
        #     for i in range(len(ncsp)):
        #     ##TODO: change this:
        #         F += inner(dot(ncsp[i][0],n), qs[ncsp[i][1]])*self.ds(ncsp[i][2])

        # #FIXME: it needs validation: for now it only works with one network
        # rcsu, rcsp = self.problem.robin_conditions(time, time_dt, theta)
        # if len(rcsu) > 0:
        #     F += dot(alphas[0]*ps[0]*n,v) * self.ds(rcsu[i])
        #     print "added a robin term"
        # Extract the left and right-hand sides (l and r)
        # (l, r) = system(F)
        l = lhs(F)
        r = rhs(F)
        # Assemble the stiffness matrix
          
        # Put all Dirichlet boundary conditions together in list.
            
        A, RHS = assemble_system(l, r, bcs)  

        print "A built"

        if self.params.direct_solver == False:
            
            # Define preconditioner form:
            pu = mu * inner(grad(u), grad(v))*dx #+ nu*E/((1.0-2.0*nu)*(1.0+nu))*inner(u,v)*dx 
            # pu = inner(sym(grad(u)), sym(grad(v)))*dx
            pp = sum(1./Q*ps[i]*qs[i]*dx + dt*theta* Ks[i]*inner(grad(ps[i]), grad(qs[i]))*dx + dt*theta*G[i]*ps[i]*qs[i]*dx
                    for i in range(AA))
            prec = pu + pp

            if nullspace==True:
            # Since there are no bc on u I need to make the preconditioner pd adding a mass matrix
                prec += inner(u, v)*dx
                prec += sum(zs[i]*rs[i]*dx for i in range(len(Z)))
            
            # Assemble preconditioner
            print "building preconditioner!!!!"
    
            P, _ = assemble_system(prec, r, bcs)
            
            if self.params.fieldsplit:
                
                sol = PETScKrylovSolver()
                sol.set_operators(A, P)
                
                
                pre_type = "hypre"
                
                # pre_type = "ml"
                
                PETScOptions().set("ksp_monitor_true_residual")
                PETScOptions().set("ksp_converged_reason")
                # change to minres if it is symmetric
                PETScOptions().set("ksp_type", "minres")
                PETScOptions().set("ksp_gmres_restart", 300)
                PETScOptions().set("pc_type", "fieldsplit")
                PETScOptions().set("pc_fieldsplit_type", "additive")
                
                PETScOptions().set("fieldsplit_0_ksp_type", "preonly")
                PETScOptions().set("fieldsplit_0_pc_type", pre_type)
                PETScOptions().set("fieldsplit_0_pc_hypre_type", "boomeramg")
                
                PETScOptions().set("fieldsplit_1_ksp_type", "preonly")
                PETScOptions().set("fieldsplit_1_pc_type", pre_type)
                PETScOptions().set("fieldsplit_1_pc_hypre_type", "boomeramg")
                
                PETScOptions().set("fieldsplit_2_ksp_type", "preonly")
                PETScOptions().set("fieldsplit_2_pc_type", pre_type)
                PETScOptions().set("fieldsplit_2_pc_hypre_type", "boomeramg")
                sol.ksp().setFromOptions()
                sol.ksp().setTolerances(atol = 1e-10, rtol = 1e-10, max_it = 400)
                
                from petsc4py import PETSc
                print "setting fieldsplit"
                isets = [PETSc.IS().createGeneral(ME.sub(i).dofmap().dofs()) for i in xrange(3)]
                sol.ksp().getPC().setFieldSplitIS(*zip(["0", "1", "2"], isets))
                vec = lambda x: as_backend_type(x).vec()
                # print "solving"
                # b = Function(ME).vector()
                # from numpy.random import randn
                # b[:] = randn(b.local_size())
                # x = Function(ME).vector()
                
                # from IPython import embed; embed()
        
                #vec(b) right hand side, x is my solution
                # sol.ksp().solve(vec(b), vec(x))
            
            else:
                
                # sol = PETScKrylovSolver("minres", "amg")
                # sol = PETScKrylovSolver("minres", "hypre_amg")
                sol = KrylovSolver("minres", "hypre_amg")
                sol.parameters.update(self.params["krylov_solver"])
                sol.set_operators(A, P)  
        else:
            
            sol = LUSolver(A, "default")

        # Time stepping

        # the iterative solver start with the initial conditions
        # for testing: random vector
        
        if self.params.testing:

            print "testing"
            U.vector()[:] = np.random.randn(U.vector().array().size)
            
        else:
            U.assign(U0)
          
        while t < T-DOLFIN_EPS:
            print len(bcs)
            log(INFO, "Solving on time interval: (%g, %g)" % (t, t+dt))

            # Assemble the right-hand side

            # # Apply Dirichlet boundary conditions
            _, RHS = assemble_system(l, r, bcs) 

            if self.params.fieldsplit:
                sol.ksp().guess_nonzero = True    
                sol.ksp().solve(vec(RHS), vec(U.vector()))
                
            else:
                niter = sol.solve(U.vector(), RHS)
                print "niter = ", niter
            # U.assign(Function(ME,x))
            # Orthogonalize RHS vector b with respect to the null space (this
            # gurantees a solution exists)

            # sol.solve(U.vector(), RHS)

            # Update previous solution
            if self.problem.params.Acceleration == True:
                U_1.assign(U0)
            
            U0.assign(U)

            # Yield current solution
            yield (U, t + dt)

            # Update time (expressions) updated automatically
            t += dt
            time.assign(t)
            time_dt.assign(t + dt)




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
    
class SimpleSolver(object):
    """This solver solves the multiple-network poroelasticity equations
    (MPET): find a vector field (the displacement) u and the network
    pressures p_a for a set of networks a = 1, ..., A such that:

        - div ( sigma(u) - sum_{a} alpha_a p_a ) = f     (1)
        c_a p_t + alpha_a div(u_t) - div K_a p_a = g_a   (2)

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

    Dirichlet: 

      u(., t) = \bar u(t) 

    Neumann:

      (sigma(u) - sum_{a} alpha_a p_a I) * n = traction

    For the continuity equations (2):

    Dirichet:

      p_a(., t) = \bar p_a(t) 
      
    Neumann

      K grad p_a(., t) * n = flux_a(t)

    Robin

     ...

    Initial conditions:

      u(x, t_0) = u_0(x)

      p_a(x, t_0) = p0_a(x) if c_a > 0
    """

    def __init__(self, problem, params=None):
        "Create solver with given MPET problem and parameters."
        self.problem = problem

        # Redefine measures based on subdomain information provided by
        # the problem
        self.ds = Measure("ds", domain=self.problem.mesh,
                          subdomain_data=self.problem.facet_domains)
        self.dx = Measure('dx', domain=self.problem.mesh)

        # Update parameters if given
        self.params = self.default_params()
        if params is not None:
            self.params.update(params)
                
    @staticmethod
    def default_params():
        "Define default solver parameters."
        params = Parameters("SimpleSolver")
        params.add("dt", 0.05)
        params.add("t", 0.0)
        params.add("T", 1.0)
        params.add("theta", 0.5)
        params.add("u_degree", 2)
        params.add("p_degree", 1)
        params.add("direct_solver", True)
        params.add(KrylovSolver.default_parameters())
        params.add(LUSolver.default_parameters())
        params.add("testing", False)
        params.add("fieldsplit", False)
        params.add("symmetric", False)
        return params

    def solve(self):

        # Extract number of networks (AA)
        AA = self.problem.params.AA

        # Create function spaces
        V = VectorElement("CG", self.problem.mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", self.problem.mesh.ufl_cell(), self.params.p_degree)

        V0 = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W0 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        ## MER: Note This syntax will be obsolete in FEniCS v1.7. The
        ## new syntax will be something like:
        MX = MixedElement([V] + [W for i in range(AA)])
        ME = FunctionSpace(self.problem.mesh, MX)

        # Create solution field(s)
        U0 = Function(ME, name="U0") # Previous solution
        u0 = split(U0)[0]
        ps0 = split(U0)[1:]
        U = Function(ME, name="U") # Current solution

        # Create trial functions
        solutions = TrialFunctions(ME)
        u = solutions[0]
        ps = solutions[1:]

        # Create test functions
        tests = TestFunctions(ME)
        v = tests[0]
        qs = tests[1:]

        # --- Define forms, material parameters, boundary conditions, etc. ---
        n = FacetNormal(self.problem.mesh)

        # Extract material parameters from problem object
        E = self.problem.params.E           # Young's modulus
        nu = self.problem.params.nu         # Poisson ratio
        alphas = self.problem.params.alphas # Biot coefficients (one per network)
        Q = self.problem.params.Q           # Compressibility coefficients (1)
        G = self.problem.params.G           # Transfer coefficents (scalar values, representing transfer from network i to j)
        Ks = self.problem.params.Ks         # Permeability tensors (one per network)

        # Extract discretization parameters
        theta = self.params.theta           # Theta-scheme parameters (0.5 CN, 1.0 Backward Euler)
        t = self.params.t
        dt = self.params.dt                 # Timestep (fixed for now)
        T = self.params.T                   # Final time

        time = Constant(t)                  # Current time
        time_dt = Constant(t+dt)            # Next time

        # um and pms represent the solutions at time t + dt*theta
        um = theta * u + (1.0 - theta) * u0
        pms = [ (theta * ps[i] + (1.0 - theta) * ps0[i]) for i in range(AA)]

        # Expression for the elastic stress tensor in terms of Young's
        # modulus E and Poisson's ratio nu
        def sigma(u):
            d = u.geometric_dimension()
            I = Identity(d)
            s = E/(1.0 + nu)*sym(grad(u)) + nu*E/((1.0-2.0*nu)*(1.0+nu))*div(u)*I
            return s

        # Define 'elliptic' right hand side at t + dt
        ff = self.problem.f(time_dt)

        # Define 'parabolic' right hand side(s) at t + theta*dt
        gg0 = self.problem.g(time)
        gg1 = self.problem.g(time_dt)
        gg = [(theta*gg1[i] + (1.0-theta)*gg0[i]) for i in range(AA)]

        # Create Dirichlet boundary conditions at t + dt Assuming that
        # boundary conditions are specified via a subdomain for now
        # (FIXME more general later)
        bcsu, bcsp = self.problem.boundary_conditions(time_dt)

        bcu = [DirichletBC(ME.sub(0), bcsu[i][0], bcsu[i][1])
               for i in range(len(bcsu))]

        bcp = [DirichletBC(ME.sub(bcsp[i][0]), bcsp[i][1], bcsp[i][2])
               for i in range(len(bcsp))]

        # Extract the given initial conditions and interpolate into
        # solution finite element spaces
        ## MER: I find the 'time' argument a little funny, but keep it
        ## if you would like!
        u_init, ps_init = self.problem.initial_conditions(time)
        u_init = interpolate(u_init, V0)
        ps_init = [interpolate(ps_init_i, W0) for ps_init_i in ps_init]

        # Update the 'previous' solution with the initial conditions
        assign(U0.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U0.sub(i+1), ps_init_i)

        # --- Define the variational forms ---

        # Elliptic part
        # a = inner(sigma(u), grad(v))*dx
        a = inner(sigma(u), sym(grad(v)))*dx
        b = - sum([alphas[i]*ps[i]*div(v) for i in range(AA)])*dx
        f = dot(ff, v)*dx

        # Parabolic part(s)
        d = sum([alphas[i]*1.0/dt*div(u - u0)*qs[i]
                 for i in range(AA)])*dx
        e = sum([Ks[i]*inner(grad(pms[i]), grad(qs[i]))
                 for i in range(AA)])*dx
        s = sum([sum([G[j]*(pms[i] - pms[j])*qs[i] for j in range(AA)])
                 for i in range(AA)]) * dx
        g = sum([gg[i]*qs[i] for i in range(AA)]) * dx

        # Combined variational form (F = 0 to be solved)
        F = a + b - f + d + e + s - g

        ## If Q is infinity, the term below drops and we do not
        ## include it in the system. Oppositely, we add this term.
        if (self.problem.params.Incompressible == False):
            c = sum([1.0/(dt*Q)*(ps[i] - ps0[i])*qs[i] for i in range(AA)])*dx
            F += c

        # FIXME: This needs testing
        #attention we must pass ds and dx of the problem!!!
        #neumann bc if there are neumann bc add them to the form
        #.....
        
        
        ncsu, ncsp = self.problem.neumann_conditions(time, time_dt, theta)
        if len(ncsu) > 0:
            for i in range(len(ncsu)):
                F += dot(ncsu[i][0]*n,v) * self.ds(ncsu[i][1])
        if len(ncsp) > 0:
            for i in range(len(ncsp)):
                F += inner(dot(ncsp[i][0],n), qs[ncsp[i][1]])*self.ds(ncsp[i][2])

        # Extract the left and right-hand sides (l and r)
        (l, r) = system(F)

        # Assemble the stiffness matrix
        print "build A"
        A = assemble(l)
        # Create linear algebra object for the right-hand side vector
        # (to be reused) for performance
        RHS = Vector()

        # Put all Dirichlet boundary conditions together in list.
        bcs = []
        for i in bcu:
            bcs += [i]
        for i in bcp:
            bcs += [i]
            
        # Apply boundary conditions to the matrix
        for bc in bcs:
            bc.apply(A)

        # Define nullspace
        
        null = self.problem.nullspace()
        xs = []
        basis = None
        for nz in null:        
            # Create vector that spans the null space and normalize
            nz = interpolate(nz, ME).vector()
            nz *= 1.0/nz.norm("l2")

            # Create null space basis object and attach to PETSc matrix
            xs += [nz]
            
        if len(xs) > 0:    
            basis = VectorSpaceBasis(xs)
            print "orthogonal = ", basis.is_orthogonal()
            print "orthonormal = ", basis.is_orthonormal()
            basis.orthonormalize()
            print "orthonormal = ", basis.is_orthonormal()
            
        # Initialize suitable linear solver
        if self.params.direct_solver:
            # Initialize LUSolver
            #FIXME
            # if basis:
            #     as_backend_type(A).set_nullspace(basis)

            sol = LUSolver(A)
            sol.parameters.update(self.params["lu_solver"])
        else:
       
        # Define preconditioner form:
            pu = inner(grad(u), grad(v))*dx
            pp = sum(1./dt*ps[i]*qs[i]*dx + Ks[i]*inner(grad(ps[i]), grad(qs[i]))*dx
                    for i in range(AA))
            prec = pu + pp
            
            # Assemble preconditioner
            print "building preconditioner!!!!"
            P = assemble(prec)
            print "done"

            for bc in bcs:
                bc.apply(P)
                
            sol = PETScKrylovSolver("gmres", "amg")
            sol.parameters.update(self.params["krylov_solver"])
            
            if basis:
                as_backend_type(A).set_nullspace(basis)
                as_backend_type(P).set_nullspace(basis)
            sol.set_operators(A, P)

        # Time stepping

        # the iterative solver start with the initial conditions
        # for testing: random vector
        
        if self.params.testing:
            U.vector()[:] = np.random.uniform(-1, 1, U.vector().array().size)
        else:
            U.assign(U0)
          
        while t < T-DOLFIN_EPS:
            # log(INFO, "Solving on time interval: (%g, %g)" % (t, t+dt))

            # Assemble the right-hand side
            assemble(r, tensor=RHS)

            # # Apply Dirichlet boundary conditions
            for bc in bcs:
                bc.apply(RHS)

            # Orthogonalize RHS vector b with respect to the null space (this
            # gurantees a solution exists)
            if basis:
                basis.orthogonalize(RHS);

            sol.solve(U.vector(), RHS)

            # Update previous solution
            U0.assign(U)

            # Yield current solution
            yield U, (t + dt)

            # Update time (expressions) updated automatically
            t += dt
            time.assign(t)
            time_dt.assign(t + dt)
            
    def solve_block(self):


        print "solve_block"
        # Extract number of networks (AA)
        #2 networks
        
        AA = self.problem.params.AA
        # Create elements
        V = VectorElement("CG", self.problem.mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", self.problem.mesh.ufl_cell(), self.params.p_degree)

        # Create function spaces
        V0 = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W0 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)
        W1 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        MX = MixedElement([V] + [W for i in range(AA)])
        ME = FunctionSpace(self.problem.mesh, MX)

        U0 = Function(ME)
        # Trial and Test functions
        v, q0, q1 = TestFunction(V), TestFunction(W0), TestFunction(W1) 
        u, p0, p1 = TrialFunction(V), TrialFunction(W0), TrialFunction(W1)
        
        u0, p00, p10 = Function(V), Function(W0), Function(W1) 
        # --- Define forms, material parameters, boundary conditions, etc. ---
        n = FacetNormal(self.problem.mesh)

        # Extract material parameters from problem object
        E = self.problem.params.E           # Young's modulus
        nu = self.problem.params.nu         # Poisson ratio
        alphas = self.problem.params.alphas # Biot coefficients (one per network)
        Q = self.problem.params.Q           # Compressibility coefficients (1)
        G = self.problem.params.G           # Transfer coefficents (scalar values, representing transfer from network i to j)
        Ks = self.problem.params.Ks         # Permeability tensors (one per network)

        # Extract discretization parameters
        theta = self.params.theta           # Theta-scheme parameters (0.5 CN, 1.0 Backward Euler)
        t = self.params.t
        dt = self.params.dt                 # Timestep (fixed for now)
        T = self.params.T                   # Final time

        time = Constant(t)                  # Current time
        time_dt = Constant(t+dt)            # Next time

        # um and pms represent the solutions at time t + dt*theta

        # Expression for the elastic stress tensor in terms of Young's
        # modulus E and Poisson's ratio nu
        def sigma(u):
            d = u.geometric_dimension()
            I = Identity(d)
            s = E/(1.0 + nu)*sym(grad(u)) + nu*E/((1.0-2.0*nu)*(1.0+nu))*div(u)*I
            return s

        # Define 'elliptic' right hand side at t + dt
        ff = self.problem.f(time_dt)

        # Define 'parabolic' right hand side(s) at t + theta*dt
        gg = self.problem.g(time_dt)


        # Create Dirichlet boundary conditions at t + dt Assuming that
        # boundary conditions are specified via a subdomain for now
        # (FIXME more general later)
        bcsu, bcsp = self.problem.boundary_conditions(time_dt)

        bcu = [DirichletBC(V, bcsu[i][0], bcsu[i][1])
               for i in range(len(bcsu))]

        bcp = [DirichletBC(W0, bcsp[0][1], bcsp[0][2]),\
               DirichletBC(W1, bcsp[0][1], bcsp[0][2]),]
        
        bcs = []
        for i in bcu:
            bcs += [i]
        for i in bcp:
            bcs += [i]
        # Extract the given initial conditions and interpolate into
        # solution finite element spaces
        ## MER: I find the 'time' argument a little funny, but keep it
        ## if you would like!
        u_init, ps_init = self.problem.initial_conditions(time)
        u_init = interpolate(u_init, V0)
        ps_init[0] = interpolate(ps_init[0], W0)
        ps_init[1] = interpolate(ps_init[1], W1)

        # Update the 'previous' solution with the initial conditions
        assign(u0, u_init)
        assign(p00, ps_init[0])
        assign(p10, ps_init[1])

        # --- Define the variational forms ---

        # bilinear forms (aaij = (i,j) block of the whole system)	
        # aa00 =  inner(sigma(u), grad(v))*dx
        aa00 =  inner(sigma(u), sym(grad(v)))*dx 
        aa01 =  alphas[0]*div(v)*p0*dx 
        aa02 =  alphas[1]*div(v)*p1*dx
    
        aa10 =  alphas[0]*div(u)*q0*dx 
        aa11 =  -Ks[0]*dt*inner(grad(p0),grad(q0))*dx - 1.0/Q*p0*q0*dx - dt*G[1]*p0*q0*dx
        aa12 =  dt*G[1]*p1*q0*dx
    
        aa20 =  alphas[1]*div(u)*q1*dx
        aa21 =  dt * G[0]*p0*q1*dx
        aa22 =  -Ks[1]*dt*inner(grad(p1), grad(q1))*dx - 1.0/Q*p1*q1*dx - dt * G[0]*p1*q1*dx
    
    
    
        # preconditioner blocks
        #adding a mass matrix so that it is not singular
        # pp00 = inner(sym(grad(u)), grad(v))*dx #+ inner(u, v) * dx
        pp00 = inner(grad(u), grad(v))*dx #+ inner(u, v) * dx
        pp11 = Ks[0]*dt*inner(grad(p0), grad(q0))*dx + 1.0/Q*p0*q0*dx + dt*G[1]*p0*q0*dx
        pp22 = Ks[1]*dt*inner(grad(p1), grad(q1))*dx + 1.0/Q*p1*q1*dx + dt*G[0]*p1*q1*dx
        
        # right-hand sides	
        L0 = dot(ff,v)*dx
        
        L1 = div(u0)*q0*dx - 1.0/Q*p00*q0*dx + dt*gg[0]*q0*dx
        
        L2 = div(u0)*q1*dx - 1.0/Q*p10*q1*dx + dt*gg[1]*q1*dx
    
        # block assemble
        print "assemble A"
        AA = block_assemble([[aa00, aa01, aa02], [aa10, aa11, aa12], [aa20, aa21, aa22]])
        print "assemble PP"
        PP = block_assemble([[pp00, 0, 0], [0, pp11, 0], [0, 0, pp22]])
        print "apply bc to PP"
        block_bc(bcs, True).apply(PP)  
        
        print "extract matrices A"
        # extract matrices
        [[A00, A01, A02],
        [A10, A11, A12], 
        [A20, A21, A22]] = AA
        print "extract matrices P"        
        [[P00, P01, P02],
        [P10, P11, P12],
        [P20, P21, P22]] = PP 

        print "assemble P using ML"        
        # construct preconditioner
        # PP = block_mat([[ML(P00), 0, 0], [0, ML(P11), 0], [0, 0, ML(P22)]])
        PP = block_mat([[AMG(P00), 0, 0], [0, AMG(P11), 0], [0, 0, AMG(P22)]])


        null = self.problem.nullspace()
        xs = []
        for nz in null:        
            # Create vector that spans the null space and normalize
            nz0 = interpolate(nz, V).vector()
            nz = block_vec([nz0, 0, 0])
            nz.allocate(AA) #from IPython import embed; embed()
            nz = 1./(nz.norm())*nz
            xs += [nz]
            
        #Gram-Schmidt orthonormalization of the null space
        basis = []
        for i in range(len(xs)):
            v = xs[i]
            for j in range(i):
                v -= xs[i].inner(xs[j])*xs[j]
            v = 1./(v.norm())*v
            basis.append(v)

        xx = block_assemble([L0, L1, L2])
        
        if self.params.testing:
            print "testing"
            xx[0][:] = np.random.randn(xx[0].size())
            xx[1][:] = np.random.randn(xx[1].size())
            xx[2][:] = np.random.randn(xx[2].size())
        else:
            xx[0][:] = u0.vector()
            xx[1][:] = p00.vector()
            xx[2][:] = p10.vector()
                
        while t< T-DOLFIN_EPS:
            log(INFO, "Solving on time interval: (%g, %g) using cbc.block" % (t, t+dt))

            # Assemble the right-hand side

            # # Apply Dirichlet boundary conditions
            bb = block_assemble([L0, L1, L2])
            
            block_bc(bcs, True).apply(AA).apply(bb)
            # def block_inner(x, y):
            #     return sum([x_i.inner(y_i) for (x_i,y_i) in zip(y,y)])
            
            for zi in basis:
                bb = bb - zi.inner(bb)*zi
                # bb = bb - zi


            Ai = MinRes2(AA, precond=PP, initial_guess=xx, tolerance=1e-6, maxiter=10000, relativeconv=True)
            x = Ai * bb
            print "MinRes iterations = ", len(Ai.residuals)
            U, P0, P1 = x
            
            # plot(project(P0,W0), mesh=self.problem.mesh)
            # interactive()
            # Orthogonalize RHS vector b with respect to the null space (this
            # gurantees a solution exists)

            # sol.solve(U.vector(), RHS)

            # Update previous solution

            u0.vector()[:] = U
            p00.vector()[:] = P0
            p10.vector()[:] = P1
            
            if self.params.testing:
                print "testing"
                xx[0][:] = np.random.random(xx[0].size())
                xx[1][:] = np.random.random(xx[1].size())
                xx[2][:] = np.random.random(xx[2].size())
            else:
                xx[0][:] = u0.vector()
                xx[1][:] = p00.vector()
                xx[2][:] = p10.vector()
            # Yield current solution
            yield (u0, p00, p10, t + dt)

            # Update time (expressions) updated automatically
            t += dt
            time.assign(t)
            time_dt.assign(t + dt)
            
    def solve_symmetric(self):

        # Extract number of networks (AA)
        AA = self.problem.params.AA
        # Create function spaces
        V = VectorElement("CG", self.problem.mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", self.problem.mesh.ufl_cell(), self.params.p_degree)

        V0 = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W0 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        Ve = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree+2)
        We = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree+2)

        nullspace = self.problem.nullspace()
        if nullspace == True:
            Z = rigid_motions(self.problem.mesh)
            R = VectorElement('R', self.problem.mesh.ufl_cell(), 0, len(Z))
            MX = MixedElement([V] + [W for i in range(AA)] + [R])
            ME = FunctionSpace(self.problem.mesh, MX)

        else:
            MX = MixedElement([V] + [W for i in range(AA)])
            ME = FunctionSpace(self.problem.mesh, MX)

        # Create solution field(s)
        U0 = Function(ME, name="U0") # Solution at t = n-1
        u0 = split(U0)[0]
        ps0 = split(U0)[1:AA+1]

        U_1 = Function(ME, name="U_1") # Solution at t = n
        u_1= split(U_1)[0]
        ps_1 = split(U_1)[1:AA+1]
        
        U = Function(ME, name="U") # Solution at t = n+1

        # Create trial functions
        solutions = TrialFunctions(ME)
        u = solutions[0]
        ps = solutions[1:AA+1]

        # Create test functions
        tests = TestFunctions(ME)
        v = tests[0]
        # q0, q1 = tests[1:]
        qs = tests[1:AA+1]

        if nullspace == True:
            zs = solutions[-1]
            rs = tests[-1]
            print "len(zs) = ", len(zs)
            
        # --- Define forms, material parameters, boundary conditions, etc. ---
        n = FacetNormal(self.problem.mesh)

        # Extract material parameters from problem object
        E = self.problem.params.E           # Young's modulus
        nu = self.problem.params.nu         # Poisson ratio
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
        alphas = self.problem.params.alphas # Biot coefficients (one per network)
        Q = self.problem.params.Q           # Compressibility coefficients (1)
        G = self.problem.params.G           # Transfer coefficents (scalar values, representing transfer from network i to j)
        Ks = self.problem.params.Ks         # Permeability tensors (one per network)

        # Extract discretization parameters
        theta = self.params.theta           # Theta-scheme parameters (0.5 CN, 1.0 Backward Euler)
        t = self.params.t
        dt = self.params.dt                 # Timestep (fixed for now)
        T = self.params.T                   # Final time

        time = Constant(t)                  # Current time
        time_dt = Constant(t+dt)            # Next time

        # um and pms represent the solutions at time t + dt*theta
        um = theta * u + (1.0 - theta) * u0
        pms = [ (theta * ps[i] + (1.0 - theta) * ps0[i]) for i in range(AA)]

        # Expression for the elastic stress tensor in terms of Young's
        # modulus E and Poisson's ratio nu
        def sigma(u):
            d = u.geometric_dimension()
            I = Identity(d)
            s = E/(1.0 + nu)*sym(grad(u)) + nu*E/((1.0-2.0*nu)*(1.0+nu))*div(u)*I
            return s

        # Define 'elliptic' right hand side at t + dt
        ff = self.problem.f(time_dt)

        # Define 'parabolic' right hand side(s) at t + theta*dt
        gg0 = self.problem.g(time)
        gg1 = self.problem.g(time_dt)
        gg = [(theta*gg1[i] + (1.0-theta)*gg0[i]) for i in range(AA)]

        # Extract the given initial conditions and interpolate into
        # solution finite element spaces
        ## MER: I find the 'time' argument a little funny, but keep it
        ## if you would like!
        u_init, ps_init = self.problem.initial_conditions(time)
        # u_init = interpolate(u_init, V)
        u_init = project(u_init, V0)
        # ps_init = [interpolate(ps_init_i, W) for ps_init_i in ps_init]
        ps_init = [project(ps_init_i, W0) for ps_init_i in ps_init]
        # Update the 'previous' solution with the initial conditions
        assign(U0.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U0.sub(i+1), ps_init_i)

        assign(U_1.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U_1.sub(i+1), ps_init_i)

        # --- Define the variational forms ---

        #Acceleration terms
               
        # Elliptic part
        a = inner(sigma(u), sym(grad(v)))*self.dx
        b = sum([- alphas[i]*ps[i]*div(v) for i in range(AA)])*self.dx
        f = dot(ff, v)*self.dx

        # Parabolic part(s)
        d =  sum([- alphas[i]*div(u - u0)*qs[i]
                 for i in range(AA)])*self.dx
        e = sum([- dt * Ks[i]*inner(grad(pms[i]), grad(qs[i]))
                 for i in range(AA)])*self.dx
        s = sum([sum([- dt * G[j]*(pms[i] - pms[j])*qs[i] for j in range(AA)])
                 for i in range(AA)]) * self.dx
        g =  sum([dt * gg[i]*qs[i] for i in range(AA)]) * self.dx

        # Combined variational form (F = 0 to be solved)
        
        F = a + b - f + d + e + s - g

        if self.problem.params.Acceleration == True:
            ddu_ddt = (u - 2*u0 + u_1)/(dt*dt)
            F += inner(ddu_ddt, v)*dx - sum([dt * div(Ks[i]*ddu_ddt)*qs[i] for i in range(AA)]) * self.dx
            
        ## If Q is infinity, the term below drops and we do not
        ## include it in the system. Oppositely, we add this term.
        if (self.problem.params.Incompressible == False):
            c = sum([-1.0/(Q)*(ps[i] - ps0[i])*qs[i] for i in range(AA)])*self.dx
            F += c

        if nullspace == True:
            F += sum(rs[i]*inner(Z[i], u)*self.dx for i in range(len(Z)))
            F += sum(zs[i]*inner(Z[i], v)*self.dx for i in range(len(Z)))

        # Create Dirichlet boundary conditions at t + dt Assuming that
        # boundary conditions are specified via a subdomain for now
        # (FIXME more general later)


        boundary_conditions_u = self.problem.boundary_conditions_u(time, time_dt, theta)
        boundary_conditions_p = self.problem.boundary_conditions_p(time, time_dt, theta)

        bcs = []
        integrals_N = []
        integrals_R = []

        for i in boundary_conditions_u:
            if 'Dirichlet' in boundary_conditions_u[i]:
                bc = DirichletBC(ME.sub(0), boundary_conditions_u[i]['Dirichlet'], self.problem.facet_domains, i)
                bcs.append(bc)
            if 'Neumann' in boundary_conditions_u[i]:
                stress = boundary_conditions_u[i]['Neumann']
                integrals_N.append(inner(stress*n,v)*self.ds(i))
            if 'Robin' in boundary_conditions_u[i]:
                r, ur = boundary_conditions[i]['Robin']
                integrals_R.append(r*inner((u - ur),v)*self.ds(i))

        for i in range(len(boundary_conditions_p)):
            bcsp = boundary_conditions_p[i]
            for j in bcsp:
                if 'Dirichlet' in bcsp[j]:
                    print "j = ", j
                    print "time = ", bcsp[j]
                    bc = DirichletBC(ME.sub(i+1), bcsp[j]["Dirichlet"], self.problem.facet_domains, j)
                    bcs.append(bc)
                if 'Neumann' in bcsp[j]:
                    flux = bcsp[j]['Neumann']
                    integrals_N.append(inner(dt * flux,n) * qs[i] * self.ds(j))
                if 'Robin' in bcsp[j]:
                    (r, pr) = bcsp[j]['Robin']
                    integrals_R.append(dt * r * (pms[i] - pr) * qs[i] * self.ds(j))

        F +=  sum(integrals_N) + sum(integrals_R)

        # FIXME: This needs testing
        #attention we must pass ds and dx of the problem!!!
        #neumann bc if there are neumann bc add them to the form
        #.....
        # ncsu, ncsp = self.problem.neumann_conditions(time, time_dt, theta)
        # if len(ncsu) > 0:
        #     for i in range(len(ncsu)):
        #         F += inner(ncsu[i][0]*n,v) * self.ds(ncsu[i][1])
        #         print assemble(1.0*self.ds(ncsu[i][1]))
        # if len(ncsp) > 0:
        #     for i in range(len(ncsp)):
        #     ##TODO: change this:
        #         F += inner(dot(ncsp[i][0],n), qs[ncsp[i][1]])*self.ds(ncsp[i][2])

        # #FIXME: it needs validation: for now it only works with one network
        # rcsu, rcsp = self.problem.robin_conditions(time, time_dt, theta)
        # if len(rcsu) > 0:
        #     F += dot(alphas[0]*ps[0]*n,v) * self.ds(rcsu[i])
        #     print "added a robin term"
        # Extract the left and right-hand sides (l and r)
        # (l, r) = system(F)
        l = lhs(F)
        r = rhs(F)
        # Assemble the stiffness matrix
          
        # Put all Dirichlet boundary conditions together in list.
            
        A, RHS = assemble_system(l, r, bcs)  

        print "A built"

        if self.params.direct_solver == False:
            
            # Define preconditioner form:
            pu = mu * inner(grad(u), grad(v))*dx #+ nu*E/((1.0-2.0*nu)*(1.0+nu))*inner(u,v)*dx 
            # pu = inner(sym(grad(u)), sym(grad(v)))*dx
            pp = sum(1./Q*ps[i]*qs[i]*dx + dt*theta* Ks[i]*inner(grad(ps[i]), grad(qs[i]))*dx + dt*theta*G[i]*ps[i]*qs[i]*dx
                    for i in range(AA))
            prec = pu + pp

            if nullspace==True:
            # Since there are no bc on u I need to make the preconditioner pd adding a mass matrix
                prec += inner(u, v)*dx
                prec += sum(zs[i]*rs[i]*dx for i in range(len(Z)))
            
            # Assemble preconditioner
            print "building preconditioner!!!!"
    
            P, _ = assemble_system(prec, r, bcs)
            
            if self.params.fieldsplit:
                
                sol = PETScKrylovSolver()
                sol.set_operators(A, P)
                
                
                pre_type = "hypre"
                
                # pre_type = "ml"
                
                PETScOptions().set("ksp_monitor_true_residual")
                PETScOptions().set("ksp_converged_reason")
                # change to minres if it is symmetric
                PETScOptions().set("ksp_type", "minres")
                PETScOptions().set("ksp_gmres_restart", 300)
                PETScOptions().set("pc_type", "fieldsplit")
                PETScOptions().set("pc_fieldsplit_type", "additive")
                
                PETScOptions().set("fieldsplit_0_ksp_type", "preonly")
                PETScOptions().set("fieldsplit_0_pc_type", pre_type)
                PETScOptions().set("fieldsplit_0_pc_hypre_type", "boomeramg")
                
                PETScOptions().set("fieldsplit_1_ksp_type", "preonly")
                PETScOptions().set("fieldsplit_1_pc_type", pre_type)
                PETScOptions().set("fieldsplit_1_pc_hypre_type", "boomeramg")
                
                PETScOptions().set("fieldsplit_2_ksp_type", "preonly")
                PETScOptions().set("fieldsplit_2_pc_type", pre_type)
                PETScOptions().set("fieldsplit_2_pc_hypre_type", "boomeramg")
                sol.ksp().setFromOptions()
                sol.ksp().setTolerances(atol = 1e-10, rtol = 1e-10, max_it = 400)
                
                from petsc4py import PETSc
                print "setting fieldsplit"
                isets = [PETSc.IS().createGeneral(ME.sub(i).dofmap().dofs()) for i in xrange(3)]
                sol.ksp().getPC().setFieldSplitIS(*zip(["0", "1", "2"], isets))
                vec = lambda x: as_backend_type(x).vec()
                # print "solving"
                # b = Function(ME).vector()
                # from numpy.random import randn
                # b[:] = randn(b.local_size())
                # x = Function(ME).vector()
                
                # from IPython import embed; embed()
        
                #vec(b) right hand side, x is my solution
                # sol.ksp().solve(vec(b), vec(x))
            
            else:
                
                # sol = PETScKrylovSolver("minres", "amg")
                # sol = PETScKrylovSolver("minres", "hypre_amg")
                sol = KrylovSolver("minres", "hypre_amg")
                sol.parameters.update(self.params["krylov_solver"])
                sol.set_operators(A, P)  
        else:
            
            sol = LUSolver(A, "default")

        # Time stepping

        # the iterative solver start with the initial conditions
        # for testing: random vector
        
        if self.params.testing:

            print "testing"
            U.vector()[:] = np.random.randn(U.vector().array().size)
            
        else:
            U.assign(U0)
          
        while t < T-DOLFIN_EPS:
            print len(bcs)
            log(INFO, "Solving on time interval: (%g, %g)" % (t, t+dt))

            # Assemble the right-hand side

            # # Apply Dirichlet boundary conditions
            _, RHS = assemble_system(l, r, bcs) 

            if self.params.fieldsplit:
                sol.ksp().guess_nonzero = True    
                sol.ksp().solve(vec(RHS), vec(U.vector()))
                
            else:
                niter = sol.solve(U.vector(), RHS)
                print "niter = ", niter
            # U.assign(Function(ME,x))
            # Orthogonalize RHS vector b with respect to the null space (this
            # gurantees a solution exists)

            # sol.solve(U.vector(), RHS)

            # Update previous solution
            if self.problem.params.Acceleration == True:
                U_1.assign(U0)
            
            U0.assign(U)

            # Yield current solution
            yield (U, t + dt)

            # Update time (expressions) updated automatically
            t += dt
            time.assign(t)
            time_dt.assign(t + dt)



    def solve_totalpressure(self):

        log(INFO,"Extract number of networks (AA)")
        AA = self.problem.params.AA

        log(INFO, "Create function spaces")
        V = VectorElement("CG", self.problem.mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", self.problem.mesh.ufl_cell(), self.params.p_degree)

        V0 = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W0 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        nullspace = self.problem.nullspace()
        
        if nullspace == True:
            Z = rigid_motions(self.problem.mesh)
            #Function or VectorFunction Space?
            # R = FunctionSpace(self.problem.mesh, 'R', 0, len(Z))
            R = VectorElement('R', self.problem.mesh.ufl_cell(), 0, len(Z))
            MX = MixedElement([V] + [W for i in range(AA+1)] + [R])
            ME = FunctionSpace(self.problem.mesh, MX)
        else:
            MX = MixedElement([V] + [W for i in range(AA+1)])
            ME = FunctionSpace(self.problem.mesh, MX)

        log(INFO,"Create solution field(s)")
        U0 = Function(ME, name="U0") # Previous solution
        u0 = split(U0)[0]#displacement
        pt0 = split(U0)[1]#total pressure
        ps0 = split(U0)[2:AA+2]#fluid pressures        
        U = Function(ME, name="U") # Current solution

        log(INFO, "Create trial functions")
        solutions = TrialFunctions(ME)
        u = solutions[0]
        pt = solutions[1]
        ps = solutions[2:AA+2]

        log(INFO, "Create test functions")
        tests = TestFunctions(ME)
        v = tests[0]
        qt = tests[1]
        qs = tests[2:AA+2]

        if nullspace == True:
            zs = solutions[-1]
            rs = tests[-1]
            

        # --- Define forms, material parameters, boundary conditions, etc. ---
        n = FacetNormal(self.problem.mesh)

        # Extract material parameters from problem object
        E = self.problem.params.E           # Young's modulus
        nu = self.problem.params.nu         # Poisson ratio
        
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
        alphas = self.problem.params.alphas # Biot coefficients (one per network)
        Q = self.problem.params.Q           # Compressibility coefficients (1)
        G = self.problem.params.G           # Transfer coefficents (scalar values, representing transfer from network i to j)
        Ks = self.problem.params.Ks         # Permeability tensors (one per network)

        # Extract discretization parameters
        theta = self.params.theta           # Theta-scheme parameters (0.5 CN, 1.0 Backward Euler)
        t = self.params.t
        dt = self.params.dt                 # Timestep (fixed for now)
        T = self.params.T                   # Final time

        time = Constant(t)                  # Current time
        time_dt = Constant(t+dt)            # Next time

        # um and pms represent the solutions at time t + dt*theta
        um = theta * u + (1.0 - theta) * u0
        pms = [ (theta * ps[i] + (1.0 - theta) * ps0[i]) for i in range(AA)]


        # Define 'elliptic' right hand side at t + dt
        ff = self.problem.f(time_dt)

        # Define 'parabolic' right hand side(s) at t + theta*dt
        gg0 = self.problem.g(time)
        gg1 = self.problem.g(time_dt)
        gg = [(theta*gg1[i] + (1.0-theta)*gg0[i]) for i in range(AA)]

        log(INFO, "BC")
        # Create Dirichlet boundary conditions at t + dt Assuming that
        # boundary conditions are specified via a subdomain for now
        # (FIXME more general later)
        bcsu, bcsp = self.problem.boundary_conditions(time_dt)

        bcu = [DirichletBC(ME.sub(0), bcsu[i][0], bcsu[i][1])
               for i in range(len(bcsu))]

        bcp = [DirichletBC(ME.sub(bcsp[i][0]), bcsp[i][1], bcsp[i][2])
               for i in range(len(bcsp))]

        log(INFO, "BC_applied")
        # Extract the given initial conditions and interpolate into
        # solution finite element spaces
        # MER: I find the 'time' argument a little funny, but keep it
        # if you would like!
        u_init, ps_init = self.problem.initial_conditions(time)
        # u_init = interpolate(u_init, V)
        # ps_init = [interpolate(ps_init_i, W) for ps_init_i in ps_init]

        log(INFO, "IC")
        u_init = project(u_init, V0)
        ps_init = [project(ps_init_i, W0) for ps_init_i in ps_init]
        log(INFO, "IC_applied")

        # Update the 'previous' solution with the initial conditions
        
        assign(U0.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U0.sub(i+1), ps_init_i)

        # --- Define the variational forms ---

        log(INFO, "Momentum equations")
        a = 2.0*mu*inner(sym(grad(u)), sym(grad(v)))*dx
        b = (pt*div(v))*dx
        f = dot(ff, v)*dx

        log(INFO,"Total pressure equation:")
        h = div(u) * qt * dx\
            -1.0/lmbda * pt*qt*dx - 1.0/lmbda * sum([alphas[i]*ps[i]*qt for i in range(AA)])*dx

        log(INFO, "Continuity equations")
        d = sum([ -1.0/lmbda*alphas[i] * (pt - pt0) * qs[i] \
                  -1.0/lmbda*alphas[i] * sum([alphas[j]*(ps[j]-ps0[j]) for j in range(AA)])*qs[i] 
                  for i in range(AA)])*dx

        e = sum([- dt * Ks[i]*inner(grad(pms[i]), grad(qs[i]))
                 for i in range(AA)])*dx

        s = sum([sum([- dt * G[j]*(pms[i] - pms[j])*qs[i] for j in range(AA)])
                 for i in range(AA)]) * dx

        g = sum([dt * gg[i]*qs[i] for i in range(AA)]) * dx

        # Combined variational form (F = 0 to be solved)

        F = a + b - f + d + e + s - g + h 

        # If Q is infinity, the term below drops and we do not
        # include it in the system. Oppositely, we add this term.
        if (self.problem.params.Incompressible == False):
            c = sum([-1.0/(Q)*(ps[i] - ps0[i])*qs[i] for i in range(AA)])*dx
            F += c


        if nullspace == True:
            log(INFO, "Assembling lagrange multipliers terms")
            F += sum(rs[i]*inner(Z[i], u)*dx for i in range(len(Z)))
            F += sum(zs[i]*inner(Z[i], v)*dx for i in range(len(Z)))

            
            # F += zs[0]*inner(Z[0], v)*dx        

        # FIXME: This needs testing
        #attention we must pass ds and dx of the problem!!!
        #neumann bc if there are neumann bc add them to the form
        #.....

        ncsu, ncsp = self.problem.neumann_conditions(time, time_dt, theta)
        if len(ncsu) > 0:
            for i in range(len(ncsu)):
                F += dot(ncsu[i][0]*n,v) * self.ds(ncsu[i][1])
        if len(ncsp) > 0:
            for i in range(len(ncsp)):
                F += inner(ncsp[i][0]*n, qs[ncsp[i][1]])*self.ds(ncsp[i][2])
        log(INFO, "term in the form all summed")
        # Introduce a term that add Robin conditions?
        
        
        # Extract the left and right-hand sides (l and r)
        log(INFO, "Extract the left and the right-hand sides")

        (l, r) = system(F)

        # Put all Dirichlet boundary conditions together in list.
        bcs = []
        for i in bcu:
            bcs += [i]
        for i in bcp:
            bcs += [i]
            
        # A = PETScMatrix()
        log(INFO, "Assembling A")
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print 'Memory usage: %s (kb)' % (mem)

        A, RHS = assemble_system(l, r, bcs) 
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print 'Memory usage: %s (kb)' % (mem)

        log(INFO, "A assambled") 
        print "A built"
        # print "is A symmetric? ", A.is_symmetric(1.0e-15)
        
        if self.params.direct_solver == False:
        
        # Define preconditioner form:
            pu = mu * inner(grad(u), grad(v))*dx
            pp = sum(alphas[i]*alphas[i]/lmbda*ps[i]*qs[i]*dx + dt*theta*Ks[i]*inner(grad(ps[i]), grad(qs[i]))*dx + (1./Q + dt*theta*G[i])*ps[i]*qs[i]*dx
                    for i in range(AA))
            ppt = pt*qt*dx
            prec = pu + pp + ppt
            
            if nullspace==True:
                # Since there are no bc on u I need to make the preconditioner pd adding a mass matrix
                prec += inner(u, v)*dx
                prec += sum(zs[i]*rs[i]*dx for i in range(len(Z)))

            # Assemble preconditioner
            log(INFO, "Assembling Preconditioner")

            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print 'Memory usage: %s (kb)' % (mem)
            P, _ = assemble_system(prec, r, bcs)
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print 'Memory usage: %s (kb)' % (mem)
            
            log(INFO, "Preconditioner assembled")

            # sol = PETScKrylovSolver("minres", "amg")
            # sol = PETScKrylovSolver("minres", "hypre_amg")
            # sol = KrylovSolver("minres", "hypre_amg")
            sol = KrylovSolver("minres", "hypre_amg")
            sol.parameters.update(self.params["krylov_solver"])
            sol.set_operators(A, P)
            # sol.set_operator(A)
            
            # Time stepping
    
            # the iterative solver start with the initial conditions
            # for testing: random vector
            
            if self.params.testing:
    
                print "testing"
                U.vector()[:] = np.random.randn(U.vector().array().size)
                
            else:
                U.assign(U0)
              
        else:           
            sol = LUSolver(A, "default")
        
        while t < T-DOLFIN_EPS:
            log(INFO, "Solving on time interval: (%g, %g)" % (t, t+dt))

            # Assemble the right-hand side

            # # Apply Dirichlet boundary conditions
            _, RHS = assemble_system(l, r, bcs) 
            
            if self.params.fieldsplit:
                sol.ksp().guess_nonzero = True    
                sol.ksp().solve(vec(RHS), vec(U.vector()))
                
            else:
                niter = sol.solve(U.vector(), RHS)
                print "niter = ", niter
            # solve(A, U.vector(), RHS)
                
            # U.assign(Function(ME,x))
            # Orthogonalize RHS vector b with respect to the null space (this
            # gurantees a solution exists)

            # sol.solve(U.vector(), RHS)

            # Update previous solution
            U0.assign(U)

            # Yield current solution
            yield (U, t + dt)

            # Update time (expressions) updated automatically
            t += dt
            time.assign(t)
            time_dt.assign(t + dt)
        
