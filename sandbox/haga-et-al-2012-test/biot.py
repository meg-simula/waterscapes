# This demo implements the two-material poroelasticity test case from
# Haga, Osnes and Langtangen, 2012. (Figure 1a)
#
# Biot's equations (ignoring acceleration): find the displacement u
# and the pore/fluid pressure p such that:
#
#             - div(sigma(u) - alpha p I) = f
#  s_0 p_t + alpha div u_t - div k grad p = g
# 
# Backward Euler in time gives for each time-step n = 1, 2, ...: 
#  
#               - div(sigma(u^n) - alpha p^n I) = f^n
# s_0 p^n + alpha div u^n  - dt div(k grad p^n) = dt g^n - s_0 p^{n-1} + alpha div u^{n-1}
#
# Denote u = u^n, p = p^n and p^{n-1} = p0, u^{n-1} = u0 for brevity
#
# Standard two-field formulation and integration by parts:
#
# < sigma(u) - alpha p I, eps(v) > - << sigma(u)*n - alpha p n, v >> = <f^n, v>
# < s_0 p + alpha div u, q > + < dt k grad p, grad q > - << dt k grad p * n, q >> = < dt g^n - s_0 p0 + alpha div u0, q>
#
# where <.,.> denotes the L^2 inner product over the domain and <<.,.>> over the boundary.
#
# In two-field formulation:
#
# * Use CG1^2 x CG1 to illustrate major instabilities;
# * Use CG2^2 x CG1 to illustrate minor local oscillation.
#
# Standard three-field formulation: introduce the fluid flux v
#
# v = k grad p
#
# Then we have:
#
#       - div(sigma(u) - alpha p I) = f
#   s_0 p_t + alpha div u_t - div v = g
#                      v - k grad p = 0
#
# Backward Euler in time: find u, v and p such that
#
#       - div(sigma(u) - alpha p I) = f
#   s_0 p + alpha div u - dt div v = dt g + s_0 p_ + alpha div u_
#                      k^-1 v - grad p = 0
#
# Three-field weak formulation: given u_, p_, find u, v, p such that
#
#   <sigma(u) - alpha p I, eps(v) > - << sigma(u)*n - alpha p n, v >> = <f^n, v>
#                                 <s_0 p + alpha div u - dt div v, q> = <dt g + s_0 p_ + alpha div u_, q>
#                         <k^-1 v, phi> + <p, div phi> - << p, v*n >> = 0


from dolfin import *

def plot_over_line(p, k):
    # Plot line
    import numpy
    import pylab
    pylab.rcParams['figure.figsize'] = 20, 15
    pylab.rcParams['lines.linewidth'] = 2
    points = numpy.linspace(0, 1, 100)
    values = [p(0.75, z) for z in points]
    pylab.plot(points, values, 'b-', label="p")
    values = [k(0.75, z) for z in points]
    pylab.plot(points, values, 'r--', label="k")
    pylab.grid(True)
    pylab.legend()
    pylab.xlabel("z")
    pylab.ylabel("p")
    pylab.show()

n = 40
mesh = UnitSquareMesh(n, n)
middle = CompiledSubDomain("x[1] > 0.25 && x[1] <= 0.75")

# Define permeability k aka hydraulic conductance
epsilon = 1.e-8
k = Expression("(x[1] > 0.25 && x[1] <= 0.75) ? epsilon : 1.0",
               epsilon=epsilon, degree=1)
#plot(k, interactive=True, mesh=mesh)

# Define other material parameters
s0 = Constant(0.0)
alpha = Constant(1.0)
dt = Constant(1.0)
mu = Constant(1.0)
lmbda = Constant(1.0)

# Elastic stress tensor
I = Identity(mesh.topology().dim())
def sigma(u):
    return 2*mu*sym(grad(u)) + lmbda*div(u)*I

# Volume force/source
f = Constant((0.0, 0.0))
g = Constant(0.0)

n = FacetNormal(mesh)
Fx = Expression("near(x[1], 1.0) ? -1.0 : 0.0", degree=1)

def three_field():

    # Define finite element spaces
    U = VectorElement("CG", mesh.ufl_cell(), 2)
    V = FiniteElement("RT", mesh.ufl_cell(), 1)
    P = FiniteElement("DG", mesh.ufl_cell(), 0)
    
    W = FunctionSpace(mesh, MixedElement(U, V, P))

    # Initial condition for (u0, p0)
    w0 = Function(W)
    (u0, _, p0) = split(w0)

    # Test and trial functions
    (u, v, p) = TrialFunctions(W)
    (y, phi, q) = TestFunctions(W)

    # Variational formulation
    F = inner(sigma(u) - alpha*p*I, sym(grad(y)))*dx \
        - dot(Fx*n, y)*ds() \
        - inner(f, y)*dx \
        + (s0*p + alpha*div(u) - dt*div(v))*q*dx \
        - (dt*g + s0*p0 + alpha*div(u0))*q*dx \
        + inner(inv(k)*v, phi)*dx + p*div(phi)*dx 
    (a, L) = system(F)

    # Define bondary conditions
    bc0 = DirichletBC(W.sub(0).sub(0), 0.0, "near(x[0], 0.0) || near(x[0], 1.0)") # No normal displacement for solid on sides
    bc1 = DirichletBC(W.sub(0).sub(1), 0.0, "near(x[1], 0.0)")                    # No normal displacement for solid on bottom
    bcv = DirichletBC(W.sub(1), (0.0, 0.0), "near(x[1], 0.0) || near(x[0], 0.0) || near(x[0], 1.0)") # No flux on sides and bottom
    #bcp = DirichletBC(W.sub(2), 0.0, "near(x[1], 1.0)")                           # Zero fluid pressure (drained) on top, natural bc
    bcs = [bc0, bc1, bcv]

    # Solve it
    w = Function(W)
    solve(a == L, w, bcs)

    # Plot fields
    (u, v, p) = w.split(deepcopy=True)
    plot(u, title="u")
    plot(v, title="v")
    plot(p, title="p")

    plot_over_line(p, k)
    interactive()
    
def two_field():
    
    # Define finite element spaces
    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V*Q)

    # Initial condition for (u0, p0)
    w0 = Function(W)
    (u0, p0) = split(w0)

    # Variational formulation
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Two-field variational formulation
    F = inner(sigma(u) - alpha*p*I, sym(grad(v)))*dx \
        - dot(Fx*n, v)*ds() \
        - inner(f, v)*dx \
        + (s0*p + alpha*div(u))*q*dx \
        + dt*inner(k*grad(p), grad(q))*dx \
        - (dt*g + s0*p0 + alpha*div(u0))*q*dx 
    (a, L) = system(F)
    
    # Define bondary conditions
    bc0 = DirichletBC(W.sub(0).sub(0), 0.0, "near(x[0], 0.0) || near(x[0], 1.0)") # No normal displacement for solid on sides
    bc1 = DirichletBC(W.sub(0).sub(1), 0.0, "near(x[1], 0.0)")                    # No normal displacement for solid on bottom
    bcp = DirichletBC(W.sub(1), 0.0, "near(x[1], 1.0)")                           # Zero fluid pressure (drained) on top
    bcs = [bc0, bc1, bcp]

    # Solve it
    w = Function(W)
    solve(a == L, w, bcs)

    # Plot fields
    (u, p) = w.split(deepcopy=True)
    plot(u, title="u")
    plot(p, title="p")

    plot_over_line(p, k)
    interactive()
    
if __name__ == "__main__":

    three_field()
    #two_field()
