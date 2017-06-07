from dolfin import *
import math

Q = 1.0
# mu = 0.333370366254687

# E = 1.0 nu = 0.3660254037844386
# lmbda = 1.0
# mu = 0.37037037037037035

#These mu and lmbda correspond to E=1.0 and to nu = 0.499998333337037
mu = 0.3333337037032922
lmbda = 100000.0

# mu = 1.0
# lmbda = 1000.0

alpha = 1.0
k = 1.0
NN = [8, 16, 32, 64]

T = 1.0
div_free = True
errL2 = []
errH1 = []

for N in NN:
    dt = 1./N
    t = Constant(0.0)

    mesh = UnitSquareMesh(N,N)
    x = SpatialCoordinate(mesh)
    
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Vp = VectorFunctionSpace(mesh, "CG", 4)
    Wp = FunctionSpace(mesh, "CG", 4)
    
    ME = MixedFunctionSpace([V] + [W] + [W] + [W])
    
    u, pt, p1, p2 = TrialFunctions(ME)
    v, qt, q1, q2 = TestFunctions(ME)
    
    U0 = Function(ME)
    U = Function(ME)
    
     
    u0, pt0, p10, p20 = split(U0)
    #divergence free solution
    if div_free == True:
        print "div_free solution"
        ux =  pi*x[0]*cos(pi*x[0]*x[1]) * t
        uy = -pi*x[1]*cos(pi*x[0]*x[1]) * t
        uex = as_vector((ux, uy))
        dotuex = as_vector(( pi*x[0]*cos(pi*x[0]*x[1]), -pi*x[1]*cos(pi*x[0]*x[1])))
    
    #not divergence free solution
    else:
        print "not div_free solution"
        ux =  pi*x[0]*cos(pi*x[0]*x[1]) * t
        uy =  pi*x[1]*cos(pi*x[0]*x[1]) * t
        uex = as_vector((ux, uy))
        dotuex = as_vector(( pi*x[0]*cos(pi*x[0]*x[1]), pi*x[1]*cos(pi*x[0]*x[1]) ))
    
    p1ex = sin(2*pi*x[0])*sin(2*pi*x[1]) * t
    dotp1ex = sin(2*pi*x[0])*sin(2*pi*x[1])
    
    p2ex = 2.0*sin(2*pi*x[0])*sin(2*pi*x[1]) * t
    dotp2ex = 2.0*sin(2*pi*x[0])*sin(2*pi*x[1])
    
    ptex = lmbda*div(uex) - alpha * p1ex - alpha * p2ex
    
    dotptex = lmbda*div(dotuex) - alpha * dotp1ex - alpha * dotp2ex
    
    I = Identity(2)
    
    f = -div(2.0*mu*sym(grad(uex)) + ptex*I)
    g1 = -Q * dotp1ex - alpha*alpha/lmbda*dotp1ex - alpha*alpha/lmbda*dotp2ex - alpha/lmbda*dotptex + div(k*grad(p1ex))
    g2 = -Q * dotp2ex - alpha*alpha/lmbda*dotp1ex - alpha*alpha/lmbda*dotp2ex - alpha/lmbda*dotptex + div(k*grad(p2ex))
    
    
    assign(U0.sub(0), project(uex, V))
    assign(U0.sub(1), project(ptex, W))
    assign(U0.sub(2), project(p1ex, W))
    assign(U0.sub(3), project(p2ex, W))
    
    F = (2.0*mu*inner(sym(grad(u)), sym(grad(v))) + pt*div(v) - dot(f,v)\
      + div(u)*qt - 1.0/lmbda*alpha*p1*qt - 1.0/lmbda*alpha*p2*qt - 1.0/lmbda*pt*qt\
      - Q * (p1-p10)*q1 - alpha*alpha/lmbda * (p1-p10)*q1 - alpha*alpha/lmbda * (p2-p20)*q1 - alpha/lmbda * (pt-pt0)*q1 - dt*k*inner(grad(p1), grad(q1)) - dt*g1*q1\
      - Q * (p2-p20)*q2 - alpha*alpha/lmbda * (p1-p10)*q2 - alpha*alpha/lmbda * (p2-p20)*q2 - alpha/lmbda * (pt-pt0)*q2 - dt*k*inner(grad(p2), grad(q2)) - dt*g2*q2)*dx
    
    l, r = system(F)
    
    t.assign(dt)
    
    bcs = [DirichletBC(ME.sub(0), project(uex, Vp), "on_boundary"),
           DirichletBC(ME.sub(2), project(p1ex, Wp), "on_boundary"),
           DirichletBC(ME.sub(3), project(p2ex, Wp), "on_boundary")]
    
    A, RHS = assemble_system(l, r, bcs)
    
    while float(t)<=T:
        solve(A, U.vector(), RHS)
        U0.assign(U)
        t.assign(float(t+dt))
        bcs = [DirichletBC(ME.sub(0), project(uex, Vp), "on_boundary"),
               DirichletBC(ME.sub(2), project(p1ex, Wp), "on_boundary"),
               DirichletBC(ME.sub(3), project(p2ex, Wp), "on_boundary")]
    
        _, RHS = assemble_system(l, r, bcs)
        
    # print "t = ", float(t)  
    t.assign(float(t-dt))
    u, pt, p1, p2 = U.split()
    #error
    L2_error_U = assemble((u-uex)**2 * dx)**.5
    H1_error_U = assemble(grad(u-uex)**2 * dx)**.5
    L2_error_P1 = assemble((p1-p1ex)**2 * dx)**.5

    errL2 += [L2_error_U]
    errH1 += [H1_error_U]

    print "||u - uh; L^2|| = {0:1.4e}".format(L2_error_U)
    print "||u - uh; H^1|| = {0:1.4e}".format(H1_error_U)
    print "||p1 - p1h; L^2|| = {0:1.4e}".format(L2_error_P1)

    # plot(u, mesh=mesh)
    # plot(pt, mesh=mesh)
    # print "t = ", float(t)
    # plot(uex, mesh=mesh)
    # plot(ptex, mesh=mesh)
    # interactive()  



ratesL2 = [(math.log(errL2[i+1]/errL2[i]))/(math.log(0.5)) for i in range(len(NN)-1)]
ratesH1 = [(math.log(errH1[i+1]/errH1[i]))/(math.log(0.5)) for i in range(len(NN)-1)]

print "ratesL2 = ", ratesL2 
print "ratesH1 = ", ratesH1   
  
  
  
