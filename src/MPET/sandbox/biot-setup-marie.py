from __future__ import division

from dolfin import *

def E_nu_to_mu_lmbda(E, nu):
    mu = E/(2*(1.0+nu))
    lmbda = (nu*E)/((1-2*nu)*(1+nu))
    return (mu, lmbda)

L = 70 # mm
N = 1000
mesh = IntervalMesh(N, 0.0, L) 

#mesh = UnitSquareMesh(n, n)
cell = mesh.ufl_cell()
dim = mesh.topology().dim()

V = VectorElement("CG", cell, 2)
Q = FiniteElement("CG", cell, 1)

W = FunctionSpace(mesh, V*Q) 

# Material parameters
E = 584.0 # Pa 
nu = 0.35 
(mu, lmbda) = E_nu_to_mu_lmbda(E, nu)

def sigma_star(u):
    I = Identity(mesh.topology().dim())
    return 2*mu*sym(grad(u)) + lmbda*div(u)*I

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Solutions at previous timestep
w_ = Function(W)
u_, p_ = split(w_)

# Material parameters
c = Constant(4.5e-10)#10.0**(-10))
#c = Constant(0.000001)
alpha = Constant(1.0)
K = Constant(1.57*10**(-5))
# K = Constant(1.0*10**(-5))

# Timestepping parameters
k = Constant(0.1)
T = 1.0

dpdt = (p - p_)/k
dudt = (u - u_)/k

theta = Constant(1.0)
um = theta*u + (1 - theta)*u_
pm = theta*p + (1 - theta)*p_

# Body sources
f = Constant((0.0,)*dim)
g = Constant(0.0)

# Essential boundary conditions
u_bar = Constant((0.0,)*dim)
p_bar = Expression("A*(1.0 - x[0]/L) + B", degree=1, A=13.0, B=0.0, L=L)

# Boundary sources
n = FacetNormal(mesh)
t = - alpha*p_bar*n
s = Constant(0.0)

# Variational formulation
F = (inner(sigma_star(um), grad(v)) - alpha*pm*div(v) - inner(f, v)
     - c*dpdt*q - alpha*div(dudt)*q - inner(K*grad(pm), grad(q)) + g*q)*dx \
     + (inner(t, v) + s*q)*ds

(a, L) = system(F)

w = Function(W)

# bcs = [DirichletBC(W.sub(0), u_bar, "near(x[0], 0.0) && on_boundary"),
#        DirichletBC(W.sub(1), p_bar, "on_boundary")]

bcs = [DirichletBC(W.sub(0), u_bar, "on_boundary"),
       DirichletBC(W.sub(1), p_bar, "on_boundary")]

time = Constant(0.0)
while (float(time) < (T + DOLFIN_EPS)):

    solve(a == L, w, bcs)
    (u, p) = w.split(deepcopy=True)
    plot(u, key="u", title="Displacement", mode="displacement")
    plot(p, key="p", title="Pressure")
    plot(-K*grad(p), key="v", title="Fluid velocity")
    
    # Update solutions
    time.assign(float(time + k))
    w_.assign(w)
    
interactive()