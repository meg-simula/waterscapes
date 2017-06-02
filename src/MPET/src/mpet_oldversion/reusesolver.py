from dolfin import *

class ReuseLUSolver(object):

    def __init__(self, A, params=None):

        # Store operator
        self.A = A

        # Store parameters
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create LU solver and update parameters
        self.solver = LUSolver(A)
        self.solver.parameters.update(self.parameters["lu_solver"])

    def default_parameters(self):
        ps = Parameters("ReuseLULinearSolver")
        ps.add(LUSolver.default_parameters())

        # FIXME: Add reuse parameters here

        return ps

    def solve(self, x, b):
        self.solver.solve(x, b)

class ReuseKrylovSolver(object):

    def __init__(self, A, P=None, params=None):

        # Store operator(s)
        self.A = A
        self.P = P

        # Store and update parameters
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create Krylov solver and update its parameters
        s = self.parameters["solver_type"]
        p = self.parameters["preconditioner"]
        self.solver = KrylovSolver(s, p)
        if self.P is not None:
            self.solver.set_operators(A, P)
        else:
            self.solver.set_operator(A)
        self.solver.parameters.update(self.parameters["krylov_solver"])

    def default_parameters(self):
        ps = Parameters("ReuseKrylovSolver")
        ps.add("solver_type", "gmres")
        ps.add("preconditioner", "amg")
        ps.add(KrylovSolver.default_parameters())

        # FIXME: Add clever reuse parameters here

        return ps

    def solve(self, x, b):
        self.solver.solve(x, b)


if __name__ == "__main__":

    # Test with Stokes
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 2)

    M = V*Q
    (u, p) = TrialFunctions(M)
    (v, q) = TestFunctions(M)

    a = (inner(grad(u), grad(v)) + div(v)*p + div(u)*q)*dx
    p = (inner(grad(u), grad(v)) + p*q)*dx

    w = Function(M)
    b = w.vector().copy()
    b[:] = 1.0

    A = assemble(a)
    solver = ReuseLUSolver(A)
    solver.solve(w.vector(), b)
    plot(w, title="w with LU")

    w = Function(M)
    solver = ReuseKrylovSolver(A)
    solver.solve(w.vector(), b)
    plot(w, title="w with Krylov")


    interactive()
