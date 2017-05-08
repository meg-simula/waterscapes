# Code example simulating diffusion of a concentration injected inside
# the brain, used for Rognes TEDxOslo 2017 slides.

"""
  c_t - D div (grad c) = f   in O 
                     c = 0   on dO
                  c(0) = 0   initial condition

Semi-discretization:

   <c_t, q> + <D grad c, grad q> - <D grad c * n, q>_{dO} = <f, q>,

Crank-Nicolson:

   <c - c^-/dt, q> + < D grad(c^m), grad q> - < D grad(c^m) * n, q> = <f^m, q>,

where 2 c^m = (c + c^-).

"""

#import numpy
import math
from dolfin import *
#from IPython import embed

f_def = """
class Source : public Expression
{
public: 

 void eval(Array<double>& values, const Array<double>& x) const
 {
  // Intracellular 
  double d = std::sqrt((x[0] - p0)*(x[0] - p0) + (x[1] - p1)*(x[1] - p1) + (x[2] - p2)*(x[2] - p2));
  if (std::abs(d) < radius) 
    values[0] = 10.0;
  else
    values[0] = 0.0;
 }

 double p0 = 118;
 double p1 = 139;
 double p2 = 106;
 double radius = 5.0;
};
"""

f = Expression(cppcode=f_def, degree=0)

# class Source(Expression):
#     def eval(self, values, x):
#         p = (118, 139, 106)
#         r = math.sqrt(sum((x[i] - p[i])**2 for i in range(3)))
#         if abs(r < 3):
#             values[0] = 1.0
#         else:
#             values[0] = 0.0
#         return values
#f = Source(degree=0)
            
mesh = Mesh("../../meshes/brain-x/whitegray.xml.gz")
markers = MeshFunction("size_t", mesh, "../../meshes/brain-x/whitegray_markers.xml.gz")

V = FunctionSpace(mesh, "DG", 0)
file = File("output/f.pvd")
file << interpolate(f, V)
#plot(f, mesh=mesh, interactive=True)

#plot(mesh)
#plot(markers, interactive=True)

# Variational formulation
Q = FunctionSpace(mesh, "CG", 1)
c = TrialFunction(Q)
q = TestFunction(Q)

c_ = Function(Q)

dt = Constant(1)
t = Constant(dt)
D = Constant(10.0)

bc = DirichletBC(Q, 0.0, "on_boundary")
cm = 1./2*(c + c_)
F0 = (c - c_)/dt*q*dx() + inner(D*grad(cm), grad(q))*dx() - f*q*dx()
F = (c - c_)/dt*q*dx() + inner(D*grad(cm), grad(q))*dx()
a, L0 = system(F0)
_, L = system(F)

# Assemble left hand side matrix
info("Assembling")
A = assemble(a)
b = Vector(mpi_comm_world(), Q.dim())

# Step in time
c = Function(Q)
c.assign(c_)
N = 50

file = File("output/c.pvd")
file << c

if False:
    solver = LUSolver(A)
    solver.parameters["reuse_factorization"] = True
else:
    solver = PETScKrylovSolver("cg", "amg")
    solver.set_operator(A)
    #solver.parameters["monitor_convergence"] = True
    
for i in range(N):
    print "Solving %d" % i

    if (i <= 20):
        # Assemble right-hand side
        assemble(L0, tensor=b)
    else:
        assemble(L, tensor=b)
        
    # Apply boundary conditions
    bc.apply(A, b)

    # Solve linear system
    solver.solve(c.vector(), b)

    # Update previous solution
    t.assign((float(t) + float(dt)))
    c_.assign(c)
    
    #plot(c, key="c")
    file << c
    
list_timings(TimingClear_keep, [TimingType_wall])
print "Success, all done!"
interactive()
