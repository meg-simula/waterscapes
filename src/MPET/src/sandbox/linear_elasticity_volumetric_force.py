from dolfin import *
from csf_pressure import *
class MyExpression(Expression):
    def __init__(self, time, domain=None, degree=None):
        self.t = time	    
    def eval(self, value, x):
    	if sqrt(x[0]**2 + x[1]**2) < 30.001:
            value[0] = 0.0
        else:
        	value[0] = 0.05


def linear_elasicity_solver():

    mesh = Mesh("../../meshes/Donut30100.xml")
    n = FacetNormal(mesh)
    facet_domains = FacetFunction("size_t", mesh, mesh.topology().dim() - 1)
    ds = Measure("ds")(domain=mesh, subdomain_data=facet_domains)
    dx = Measure("dx")(domain=mesh)
    rV = 30.0
    rS = 100.0



    allboundary = CompiledSubDomain("on_boundary")
    ventricles = CompiledSubDomain("on_boundary & x[0]*x[0]+x[1]*x[1] <= rV*rV + 1e-9", rV = rV)
    skull = CompiledSubDomain("on_boundary & x[0]*x[0]+x[1]*x[1] >= rV*rV", rV = rV)

    allboundary.mark(facet_domains, 1)
    ventricles.mark(facet_domains, 2) 
    skull.mark(facet_domains, 3) 

    nullspace = True
    # Function Space, Functions, and time
    if nullspace == False:         
        
        V = VectorFunctionSpace(mesh, "CG", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        U = Function(V)
    else:

        V = VectorElement("CG", mesh.ufl_cell(), 1)
        W = VectorElement("R", mesh.ufl_cell(), 0, 3)
    	ME = MixedElement([V] + [W])
    	MX = FunctionSpace(mesh, ME)
    	u, p = TrialFunctions(MX)
        v, q = TestFunctions(MX) 
        U = Function(MX)


    time = Constant(0.0)
    t = 0.0 
    dt = 0.1
    T = 10.0 
    # Data
    nu = 0.35
    E = 500.0

    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    mu = E/(2.0*(1.0+nu))

    ai = [-0.0345, -0.0511, -0.0267, -0.0111, -0.0013, 0.0050,0.0027, 0.0061]
    bi = [0.1009, 0.0284,-0.0160, -0.0070, -0.0174, -0.0041, -0.0041, 0.0005]
    c0 = 102.3530
    Hg2Pa = 133.3

    pb = Hg2Pa * c0 * (1 + sum( [ai[i]*cos(2*pi*(i+1)*time) + bi[i]*sin(2*pi*(i+1)*time) for i in range(8)])) - 100*Hg2Pa
    #pb = Expression("Hg2Pa*(A1*sin(2.0*pi*t))", Hg2Pa=133.3, A1 = 40.0, t=time, domain=mesh, degree=3) 

    #Vx = Expression("(x[0]*x[0] + x[1]*x[1] - rV*rV)/(rS*rS - rV*rV)*A", rV=rV, rS=rS, A = 0.05, domain=mesh, degree=2)
    #Vx = Expression("exp(sqrt(x[0]*x[0] + x[1]*x[1])-rV)/exp(rS - rV)*A", rV=rV, rS=rS, A = 0.05, domain=mesh, degree=2)
    Vx = MyExpression(time, domain=mesh, degree=1)
    p_CSF = 4.0
    pv = Expression("Hg2Pa*A1*sin(2.0*pi*t)", Hg2Pa=133.3, A1 = p_CSF, t=time, domain=mesh, degree=3) 
    #pv = Constant(0.0*Hg2Pa)
    f = -grad(pb*Vx)

    #Forms
    sigma = 2*mu*sym(grad(u)) + lmbda*div(u)*Identity(2)

    a = inner(sigma, grad(v))*dx

    L = inner(f,v)*dx - inner(pv*n,v)*ds
    
    if nullspace==False:
        bc = [DirichletBC(V, Constant((0.0,0.0)), skull)]
    else:
        bc = []
        Z_transl = [Constant((1, 0)), Constant((0, 1))]
        Z_rot = [Expression(('x[1]', '-x[0]'), degree = 1, domain=mesh)]
        Z = Z_transl + Z_rot
        a += -sum(p[i]*inner(v, Z[i])*dx for i in range(len(Z)))\
             -sum(q[i]*inner(u, Z[i])*dx for i in range(len(Z))) 


    filef = File("results/linearelasticity_nullspace_%s"%nullspace + "_CSFAmplitude_%04.05e"%p_CSF + "_mmHg/f.pvd")
    fileU = File("results/linearelasticity_nullspace_%s"%nullspace + "_CSFAmplitude_%04.05e"%p_CSF + "_mmHg/U.pvd")
    while t < T-DOLFIN_EPS:

        #f = project(-grad(pb*Vx), V)
        plot(f, key="f", title="f", mesh=mesh)
        #plot(pb*Vx, key="pb", title="pb*Vx")
        print "pb = ", float(pb)
        #a = inner(sigma, grad(v))*dx  
        #L = inner(f,v)*dx - inner(pv*n,v)*ds(2)
        solve(a==L, U, bc)
        t += dt
        time.assign(t)
        if nullspace == True:
            fileU << U.split(deepcopy=True)[0]
        else:
        	fileU<<U

if __name__ == "__main__":

    linear_elasicity_solver()