#HB test
from dolfin import *

class Permeability(Expression):
    def eval(self, v, x):
        if x[1] >= 1.0/4.0 and x[1] <= 3.0/4.0 : v[0] = 1.0e-8
        else: v[0] = 1.0
        
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1], 1.0))

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], 1.0))
    
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1], 0.0))
    
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], 0.0))

"""

Test case from Berdal Haga PhD thesis, paper III.
\Gamma_0 = left + right + bottom
\Gamma1 = top

"""
mesh = UnitSquareMesh(64,64)
n = FacetNormal(mesh)

top = Top()
right = Right()
bottom = Bottom()
left = Left()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
top.mark(boundaries, 1)
right.mark(boundaries, 2)
bottom.mark(boundaries, 3)
left.mark(boundaries, 4)

ds = Measure("ds")(domain=mesh, subdomain_data=boundaries)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
ME = MixedFunctionSpace([V] + [W])

U0 = Function(ME) 
u0 = split(U0)[0]
p0 = split(U0)[1]

U = Function(ME) # Current solution

# Create trial functions
solutions = TrialFunctions(ME)
u = solutions[0]
p = solutions[1]

# Create test functions
tests = TestFunctions(ME)
v = tests[0]
q = tests[1]
# --- Define forms, material parameters, boundary conditions, etc. ---

# Extract material parameters from problem object
mu = 1.0
lmbda = 1.0
alpha = 1.0
# K = 1.0
K = Permeability()
Q = 1.0

# Extract discretization parameters
theta = 0.5
t = 0.0
dt = 0.01
T = 1.0

t0 = Constant(t)
t1 = Constant(t+dt)


# Expression for the elastic stress tensor in terms of Young's
# modulus E and Poisson's ratio nu
def sigma(u):
    d = u.geometric_dimension()
    I = Identity(d)
    sigma = lmbda*div(u)*I + 2*mu*sym(grad(u))
    return sigma

# Define 'elliptic' right hand side at t + dt

# Create Dirichlet boundary conditions at t + dt

# Extract the given initial conditions and interpolate into
# solution finite element spaces
u_init = interpolate(Constant((0.0, 0.0)), V)
p_init = interpolate(Constant(0.0), W)

# Update the 'previous' solution with the initial conditions
assign(U0.sub(0), u_init)
assign(U0.sub(1), p_init)

# --- Define the variational forms ---

# Elliptic part
um = theta * u + (1.0 - theta)*u0
pm = theta * p + (1.0 - theta)*p0

#Boundary condition on u only on the normal component on \Gamma0
bcu1 = DirichletBC(ME.sub(0).sub(0), Constant(0.0), "on_boundary & near(x[0], 0.0)")
bcu2 = DirichletBC(ME.sub(0).sub(1), Constant(0.0), "on_boundary & near(x[1], 0.0)")
bcu3 = DirichletBC(ME.sub(0).sub(0), Constant(0.0), "on_boundary & near(x[0], 1.0)")
bcp = DirichletBC(ME.sub(1), Constant(0.0), "on_boundary & near(x[1], 1.0)")

bc = [bcu1, bcu2, bcu3, bcp]

sigman = dot(sigma(u), n)    
eq1 = inner(sigma(u), grad(v))*dx - alpha*p*div(v)*dx - dot(n,v)*ds(1)
eq1 += - sigman[1]*v[1]*ds(2) - sigman[0]*v[0]*ds(3) - sigman[1]*v[1]*ds(4)  
eq2 = 1.0/(Q*dt)*(p-p0)*q*dx + alpha/dt*div(u-u0)*q*dx + dot(K*grad(pm), grad(q))*dx


# Combined variational form (F = 0 to be solved)
F = eq1 + eq2

# Extract the left and right-hand sides (l and r)
(l, r) = system(F)

# Assemble the stiffness matrix
A = assemble(l)

# Create linear algebra object for the right-hand side vector
# (to be reused) for performance
RHS = Vector() 


# Time stepping

while t <= T:
    log(INFO, "Solving on time interval: (%g, %g)" % (t, t+dt))

    # Assemble the right-hand side
    assemble(r, tensor=RHS)

    # # Apply Dirichlet boundary conditions
    for bbcc in bc:
        bbcc.apply(A, RHS)
    

    #Iterative solver:
    solve(A, U.vector(), RHS)
    # # Update previous solution conditions
    U0.assign(U)

    # Yield current solution
    # Update time (expressions) updated automatically

    t += dt
    t0.assign(t)
    t1.assign(t + dt)
    
plot(U.split()[0][0], mesh)
interactive()
plot(U.split()[0][1], mesh)
interactive()
plot(U.split()[1], mesh)
interactive()
