#from time import time

from cbcpost import ParamDict, Parameterized
# from block import *
# from block.iterative import *
# from block.algebraic.petsc import *
from dolfinimport import *
import numpy as np
from block import *
from block.algebraic.petsc import *
from block.iterative import *
# from dolfin import *
# from block.dolfin_util import *
# import numpy
# from cbcflow.schemes.utils import *


# class SimpleSolver(Parameterized):
class SimpleSolver():
    def __init__(self, problem, params=None):
        "Create solver with given MPET problem and parameters."
        # Parameterized.__init__(self, params)
        self.problem = problem
        self.ds = Measure("ds")(domain=problem.mesh, subdomain_data=problem.facet_domains)
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
        params.add("p_degree", 2)
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
        V = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        ## MER: Note This syntax will be obsolete in FEniCS v1.7. The
        ## new syntax will be something like:
        ## MX = MixedElement([V] + [W for i in range(AA)])
        ## ME = FunctionSpace(MX)
        ME = MixedFunctionSpace([V] + [W for i in range(AA)])

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
        u_init = interpolate(u_init, V)
        ps_init = [interpolate(ps_init_i, W) for ps_init_i in ps_init]

        # Update the 'previous' solution with the initial conditions
        assign(U0.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U0.sub(i+1), ps_init_i)

        # --- Define the variational forms ---

        # Elliptic part
        a = inner(sigma(u), grad(v))*dx
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
                F += inner(ncsp[i][0]*n, qs[ncsp[i][1]])*self.ds(ncsp[i][2])

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
            pp = sum(ps[i]*qs[i]*dx + Ks[i]*inner(grad(ps[i]), grad(qs[i]))*dx
                    for i in range(AA))
            prec = pu + pp
            
            # Assemble preconditioner
            print "building preconditioner..."
            P = assemble(prec)
            for bc in bcs:
                bc.apply(P)

            sol = PETScKrylovSolver("gmres", "amg")
            sol.parameters.update(self.params["krylov_solver"])
            
        
            if basis:
                as_backend_type(A).set_nullspace(basis)
                as_backend_type(P).set_nullspace(basis)
            sol.set_operators(A, P)
            # sol.set_operator(A)

        # Time stepping

        # the iterative solver start with the initial conditions
        # for testing: random vector
        
        if self.params.testing:
            U.vector()[:] = np.random.uniform(-1, 1, U.vector().array().size)
        else:
            U.assign(U0)
          
        while t < T:
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
        # Create function spaces

        V = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W0 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)
        W1 = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)
        ME = MixedFunctionSpace([V] + [W0] + [W1])

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
        u_init = interpolate(u_init, V)
        ps_init[0] = interpolate(ps_init[0], W0)
        ps_init[1] = interpolate(ps_init[1], W1)

        # Update the 'previous' solution with the initial conditions
        assign(u0, u_init)
        assign(p00, ps_init[0])
        assign(p10, ps_init[1])

        # --- Define the variational forms ---

        # bilinear forms (aaij = (i,j) block of the whole system)	
        aa00 =  inner(sigma(u), grad(v))*dx 
        aa01 =  alphas[0]*div(v)*p0*dx 
        aa02 =  alphas[1]*div(v)*p1*dx
    
        aa10 =  alphas[0]*div(u)*q0*dx 
        aa11 =  -Ks[0]*dt*inner(grad(p0),grad(q0))*dx - 1.0/Q*p0*q0*dx - dt*G[1]*p0*q0*dx
        aa12 =  dt*G[1]*p1*q0*dx
    
        aa20 =  alphas[1]*div(u)*q1*dx
        aa21 =  dt * G[0]*p0*q1*dx
        aa22 =  -Ks[1]*dt*inner(grad(p1), grad(q1))*dx - 1.0/Q*p1*q1*dx - dt * G[0]*p1*q1*dx
    
    
    
        # preconditioner blocks	
        pp00 = inner(sym(grad(u)), grad(v))*dx 
        pp11 = Ks[0]*dt*inner(grad(p0), grad(q0))*dx + 1.0/Q*p0*q0*dx + dt*G[1]*p0*q0*dx
        pp22 = Ks[1]*dt*inner(grad(p1), grad(q1))*dx + 1.0/Q*p1*q1*dx + dt*G[0]*p1*q1*dx
        
        # right-hand sides	
        L0 = dot(ff,v)*dx
        
        L1 = div(u0)*q0*dx - 1.0/Q*p00*q0*dx + dt*gg[0]*q0*dx
        
        L2 = div(u0)*q1*dx - 1.0/Q*p10*q1*dx + dt*gg[1]*q1*dx
    
        # block assemble
        AA = block_assemble([[aa00, aa01, aa02], [aa10, aa11, aa12], [aa20, aa21, aa22]])
        PP = block_assemble([[pp00, 0, 0], [0, pp11, 0], [0, 0, pp22]])
 
        block_bc(bcs, True).apply(PP)  
        
        # extract matrices
        [[A00, A01, A02],
        [A10, A11, A12], 
        [A20, A21, A22]] = AA
        
        [[P00, P01, P02],
        [P10, P11, P12],
        [P20, P21, P22]] = PP 
        
        # construct preconditioner
        PP = block_mat([[ML(P00), 0, 0], [0, ML(P11), 0], [0, 0, ML(P22)]])


        xx = block_assemble([L0, L1, L2])
        xx[0][:] = np.random.random(xx[0].size())
        xx[1][:] = np.random.random(xx[1].size())
        xx[2][:] = np.random.random(xx[2].size())
          
        while t < T:
            log(INFO, "Solving on time interval: (%g, %g) using cbc.block" % (t, t+dt))

            # Assemble the right-hand side

            # # Apply Dirichlet boundary conditions
            bb = block_assemble([L0, L1, L2])
            block_bc(bcs, True).apply(AA).apply(bb)             
            
            Ai = MinRes(AA, precond=PP, tolerance=1e-6, maxiter=10000, relativeconv=True)
            # from IPython import embed; embed()
            x = Ai * bb
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
            
            # Yield current solution
            yield (u0, p00, p10, t + dt)

            # Update time (expressions) updated automatically
            t += dt
            time.assign(t)
            time_dt.assign(t + dt)
            
    def solve_symmetric(self):


        # Extract number of networks (AA)
        AA = self.problem.params.AA
        print AA
        # Create function spaces
        V = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        ## MER: Note This syntax will be obsolete in FEniCS v1.7. The
        ## new syntax will be something like:
        ## MX = MixedElement([V] + [W for i in range(AA)])
        ## ME = FunctionSpace(MX)
        ME = MixedFunctionSpace([V] + [W for i in range(AA)])

        # Create solution field(s)
        U0 = Function(ME, name="U0") # Previous solution
        u0 = split(U0)[0]
        ps0 = split(U0)[1:]
        p00, p01 = split(U0)[1:]
        U = Function(ME, name="U") # Current solution

        # Create trial functions
        solutions = TrialFunctions(ME)
        u = solutions[0]
        p0, p1 = solutions[1:]
        ps = solutions[1:]

        # Create test functions
        tests = TestFunctions(ME)
        v = tests[0]
        q0, q1 = tests[1:]
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
        u_init = interpolate(u_init, V)
        ps_init = [interpolate(ps_init_i, W) for ps_init_i in ps_init]

        # Update the 'previous' solution with the initial conditions
        assign(U0.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U0.sub(i+1), ps_init_i)

        # --- Define the variational forms ---

        # Elliptic part
        a = inner(sigma(u), grad(v))*dx
        b = sum([alphas[i]*ps[i]*div(v) for i in range(AA)])*dx
        f = dot(ff, v)*dx

        # Parabolic part(s)
        d =  sum([alphas[i]*div(u - u0)*qs[i]
                 for i in range(AA)])*dx
        e = sum([- dt * Ks[i]*inner(grad(pms[i]), grad(qs[i]))
                 for i in range(AA)])*dx
        s = sum([sum([- dt * G[j]*(pms[i] - pms[j])*qs[i] for j in range(AA)])
                 for i in range(AA)]) * dx
        g =  sum([dt * gg[i]*qs[i] for i in range(AA)]) * dx

        # Combined variational form (F = 0 to be solved)
        F = a + b - f + d + e + s - g

        ## If Q is infinity, the term below drops and we do not
        ## include it in the system. Oppositely, we add this term.
        if (self.problem.params.Incompressible == False):
            c = sum([-1.0/(Q)*(ps[i] - ps0[i])*qs[i] for i in range(AA)])*dx
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
                F += inner(ncsp[i][0]*n, qs[ncsp[i][1]])*self.ds(ncsp[i][2])

        # Extract the left and right-hand sides (l and r)
        # (l, r) = system(F)
        l = lhs(F)
        r = rhs(F)
        # Assemble the stiffness matrix
          
        # Put all Dirichlet boundary conditions together in list.
        bcs = []
        for i in bcu:
            bcs += [i]
        for i in bcp:
            bcs += [i]
            
        A, RHS = block_assemble(l, r, bcs)  
        print "A built"
       
    # Define preconditioner form:
        # Pu = inner(sym(grad(u)), grad(v))*dx 
        # P0 = ps[0]*qs[0]*dx + dt * Ks[0]*inner(grad(ps[0]), grad(qs[0]))*dx + dt*G[0]*ps[0]*qs[0]*dx #+\
        # P1 = ps[1]*qs[1]*dx + dt * Ks[1]*inner(grad(ps[1]), grad(qs[1]))*dx + dt*G[1]*ps[1]*qs[1]*dx
        # 
        # prec = Pu + P0 + P1
        Fp = a + b + c - f + d + e + s - g
        lp = lhs(Fp)
        rp = rhs(Fp)
        PP, _ = block_assemble(lp, rp, bcs)
        print "building preconditioner..."
        
        # PP = block_assemble([[Pu, 0, 0], [0, P0, 0], [0, 0, P1]])
        # block_bc(bcs, True).apply(PP)  

        [[P00, P01, P02],
        [P10, P11, P12],
        [P20, P21, P22]] = PP
        
        PP = block_mat([[ML(P00),           0,           0],
                       [           0, ML(P11),           0],
                       [           0,           0, ML(P22)]])
        
        [[A00, A01, A02],
        [A10, A11, A12],
        [A20, A21, A22]] = A
            

              
                        
        # Time stepping

        # the iterative solver start with the initial conditions
        # for testing: random vector
        
        if self.params.testing:

            xx = RHS.copy()
            xx[0][:] = np.random.random(xx[0].size())
            xx[1][:] = np.random.random(xx[1].size())
            xx[2][:] = np.random.random(xx[2].size())
            print "testing"
            # U.vector()[:] = np.random.random(U.vector().array().size)
        else:
            U.assign(U0)
          
        while t < T:
            # log(INFO, "Solving on time interval: (%g, %g)" % (t, t+dt))

            # Assemble the right-hand side

            # # Apply Dirichlet boundary conditions
            # _, RHS = block_assemble(l, r, bcs) 

            Ai = MinRes(A, precond=PP, initial_guess=xx, tolerance=1e-6, maxiter=10000, relativeconv=True)
            # from IPython import embed; embed()
            x = Ai * RHS
            print "you are here"
            # U.assign(Function(ME,x))
            # Orthogonalize RHS vector b with respect to the null space (this
            # gurantees a solution exists)

            # sol.solve(U.vector(), RHS)

            # Update previous solution
            U0.assign(U)

            # Yield current solution
            yield (t + dt)

            # Update time (expressions) updated automatically
            t += dt
            time.assign(t)
            time_dt.assign(t + dt)

    def solve_totalpressure(self):

        # Extract number of networks (AA)
        AA = self.problem.params.AA

        # Create function spaces
        V = VectorFunctionSpace(self.problem.mesh, "CG", self.params.u_degree)
        W = FunctionSpace(self.problem.mesh, "CG", self.params.p_degree)

        ## MER: Note This syntax will be obsolete in FEniCS v1.7. The
        ## new syntax will be something like:
        ## MX = MixedElement([V] + [W for i in range(AA)])
        ## ME = FunctionSpace(MX)
        ME = MixedFunctionSpace([V] + [W for i in range(AA+1)])

        # Create solution field(s)
        U0 = Function(ME, name="U0") # Previous solution
        u0 = split(U0)[0]#displacement
        pt0 = split(U0)[1]#total pressure
        ps0 = split(U0)[2:]#fluid pressures        
        U = Function(ME, name="U") # Current solution

        # Create trial functions
        solutions = TrialFunctions(ME)
        u = solutions[0]
        pt = solutions[1]
        ps = solutions[2:]

        # Create test functions
        tests = TestFunctions(ME)
        v = tests[0]
        qt = tests[1]
        qs = tests[2:]

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
        u_init = interpolate(u_init, V)
        ps_init = [interpolate(ps_init_i, W) for ps_init_i in ps_init]

        # Update the 'previous' solution with the initial conditions
        assign(U0.sub(0), u_init)
        for (i, ps_init_i) in enumerate(ps_init):
            assign(U0.sub(i+1), ps_init_i)

        # --- Define the variational forms ---

        # Momentum equations
        a = 2*mu*(sym(grad(u)), sym(grad(v)))*dx
        b = (pt*div(v))*dx
        f = dot(ff, v)*dx

        # Total pressure equation:
        h = div(u)*qt * dx - 1.0/lmbda * pt*qt*dx \
          + 1.0/lmbda * sum([alphas[i]*ps[i]*qt for i in range(AA)])*dx
        
        # Continuity equations
        d = sum([ 1./lmbda*alphas[i] * (pt - pt0) * qs[i] +\
                  1./lmbda*alphas[i] * sum([alphas[j]*(ps[j]-ps0[j]) for j in range(AA)])*qs[i] 
                  for i in range(AA)])*dx

        e = sum([dt * Ks[i]*inner(grad(pms[i]), grad(qs[i]))
                 for i in range(AA)])*dx

        s = sum([sum([dt * G[j]*(pms[i] - pms[j])*qs[i] for j in range(AA)])
                 for i in range(AA)]) * dx

        g = sum([dt * gg[i]*qs[i] for i in range(AA)]) * dx

        # Combined variational form (F = 0 to be solved)
        F = a + b - f + d + e + s - g

        ## If Q is infinity, the term below drops and we do not
        ## include it in the system. Oppositely, we add this term.
        if (self.problem.params.Incompressible == False):
            c = sum([-1.0/(Q)*(ps[i] - ps0[i])*qs[i] for i in range(AA)])*dx
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
                F += inner(ncsp[i][0]*n, qs[ncsp[i][1]])*self.ds(ncsp[i][2])

        # Extract the left and right-hand sides (l and r)
        (l, r) = system(F)

        # Assemble the stiffness matrix
        print "build A"
        A = assemble(l)
        print "is A symmetric? ", A.is_symmetric(1e-6)
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
            if basis:
                as_backend_type(A).set_nullspace(basis)

            sol = LUSolver(A)
            sol.parameters.update(self.params["lu_solver"])
        else:
       
        # Define preconditioner form:
            pu = inner(grad(u), grad(v))*dx
            pp = sum(ps[i]*qs[i]*dx + dt * Ks[i]*inner(grad(ps[i]), grad(qs[i]))*dx
                    for i in range(AA))
            prec = pu + pp
            
            # Assemble preconditioner
            print "building preconditioner..."
            P = assemble(prec)
            for bc in bcs:
                bc.apply(P)

            sol = PETScKrylovSolver("gmres", "amg")
            sol.parameters.update(self.params["krylov_solver"])
            
        
            if basis:
                as_backend_type(A).set_nullspace(basis)
                as_backend_type(P).set_nullspace(basis)
            sol.set_operators(A, P)
            # sol.set_operator(A)

        # Time stepping

        # the iterative solver start with the initial conditions
        # for testing: random vector
        
        if self.params.testing:
            U.vector()[:] = np.random.random(U.vector().size())
        else:
            U.assign(U0)
          
        while t < T:
            log(INFO, "Solving on time interval: (%g, %g)" % (t, t+dt))

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
        
        # sol.ksp().view()

#class BlockSolver(Parameterized):
#
#     def __init__(self, problem, params=None):
#         Parameterized.__init__(self, params)
#
#         self.problem = problem
#
#     @classmethod
#     def default_params(cls):
#         params = ParamDict(
#             # Time parameters:
#             start_timestep = 0,
#             dt=0.1,
#             T0=0.0,
#             T=1.0,
#             u_degree = 2,
#             p_degree = 1,
#             theta = 0.5,
#             )
#         return params
#
#
#     # Function spaces, elements
#     def solve(self, problem):
#
#     mesh = problem.mesh
#     dim = mesh.topology().dim()
#
#     V = VectorFunctionSpace(mesh, "CG", params.u_degree)
#     W = FunctionSpace(mesh, "CG", params.p_degree)
#
#     u, v = TrialFunction(V), TestFunction(V)
#     ps, qs = TrialFunction(W), TestFunction(W)
#
#     u0 = Function(V)
#     p0 = Function(W)
#     u1 = Function(V)
#     p1 = Function(W)
#
#     #========
#     # Define forms, material parameters, boundary conditions, etc.
#
#     ### Material parameters from promblem.params
#
#     E = problem.params.E
#     nu = problem.params.nu
#
#     alphas = problem.parms.alphas
#     Q = problem.params.Q
#     Ks = problem.params.Ks
#     theta = params.theta
#
#
#     def sigma(u):
#         return E/(1 + nu) * sym(grad(u)) + nu * E /((1 - 2*nu)*(1 + nu)) * div(u) *Identity(dim))
#
#     def b(w, r):
#         return - alphas * r * div(w)
#
# #    def f(u, p)
# #        return -div(sigma(u1) + alphas * grad(p1)
# #    def g(u, p, du, dp)
# #        return 1/Q * dp * qs * dx + alphas * div(du1) * qs * dx + dot(Ks * grad(ps1), grad(qs)) *dx
#
#     a00 = inner(sigma(u), grad(v)) * dx
#     a01 = b(v, ps) * dx
#     a10 = -b(u, qs) * dx
#     a11 = ( 1/(Q*dt) * ps * qs + theta * inner( Ks * grad(ps), grad(qs)) ) * dx
#
#     f = -inner(div(sigma(u1)), v) + alphas * inner(grad(p1), v) #f, g, bc and ic  must be defined in the MPET class
#
#     L0 = dot(f,  v) * dx
#     L1 = -1/dt * b(u0, qs) * dx + 1/(Q*dt) * p0 * q0 - (1-theta) * inner( Ks * grad(ps), grad(qs)) ) * dx + (theta * g1 + (1-theta) * g0) * dx
#
#
#     # Create boundary conditions.
#
#     boundary = BoxBoundary(mesh)
#
#     c = 0.25
#     h = mesh.hmin()
#     fluid_source_domain = CompiledSubDomain('{min}<x[0] && x[0]<{max} && {min}<x[1] && x[1]<{max}'
#                                              .format(min=c-h, max=c+h))
#     topload_source      = Expression("-sin(2*t*pi)*sin(x[0]*pi/2)/3", t=0)
#
#     bc_u_bedrock        = DirichletBC(V,            [0]*dim,        boundary.bottom)
#     bc_u_topload        = DirichletBC(V.sub(dim-1), topload_source, boundary.top)
#     bc_p_drained_source = DirichletBC(W,            0,              fluid_source_domain)
#
#     bcs = block_bc([[bc_u_topload, bc_u_bedrock], [bc_p_drained_source]], True)
#
#     # Assemble the matrices and vectors
#
#     AA = block_assemble([[a00, a01],
#                          [a10, a11]])
#
#     rhs_bc = bcs.apply(AA)
#
#
#     t = 0.0
#     x = None
#     while t <= T:
#         print "Time step %f" % t
#
#         topload_source.t = t
#         bb = block_assemble([0, L1])
#         rhs_bc.apply(bb)
#
#         x = AAinv * bb
#
#         U,P = x
#         u = Function(V, U)
#         p = Function(W, P)
#
#         plot(u, key='u', title='u', mode='displacement')
#         plot(p, key='p', title='p')
#
#         u_prev.vector()[:] = U
#         p_prev.vector()[:] = P
#         t += float(dt)
#
# #    interactive()
# #    print "Finished normally"
#     return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(dim)
