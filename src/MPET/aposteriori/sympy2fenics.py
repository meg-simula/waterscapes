"""Scalar, vector, and matrix symbolic calculus using sympy with an interface
to generate FEniCS expressions.

By Lizao Li <lzlarryli@gmail.com>.

Examples:

    >>> import sympy2fenics as sf

    >>> f = sf.str2sympy('sin(x)')            # a scalar function in 1D
    >>> u = sf.str2sympy('(sin(x), sin(y))')  # a vector function in 2D
    >>> w = sf.str2sympy('((x,y),(x,z))')     # a matrix funciton in 2D
    >>> v = sf.str2sympy('sin(x)*sin(y)')     # a scalar function in 2D
    >>> print(sf.div(w))                      # divergence of w
    Matrix([[2], [1]])
    >>> print(sf.epsilon(u))                  # symmetric gradient of u
    Matrix([[1.0*cos(x), 0], [0, 1.0*cos(y)]])
    >>> print(sf.sym(sf.grad(u.transpose()))) # symmetric gradient of u
    Matrix([[1.0*cos(x), 0], [0, 1.0*cos(y)]])
    >>> sf.sympy2exp(w)                       # to FEniCS expression string
    (('x[0]', 'x[1]'), ('x[0]', 'x[2]'))
"""

from sympy import symbols, printing, sympify, Matrix

# pylint: disable=invalid-name

def str2sympy(expression):
    """Create sympy scalar-, vector-, or matrix-expression from a string.

    Args:
        expression (str): Formula as a string

    Returns:
        sympy.expr.Expr: sympy expression for further manipulation

    Examples:

        Variables (x,y,z) are reserved and used for automatic dimension
        inference.

            >>> f = str2sympy('sin(x)') # a scalar function in 1D
            >>> g = str2sympy('(sin(x), sin(y))') # a vector function in 2D
            >>> h = str2sympy('((x,y),(x,z))') # a matrix funciton in 2D
            >>> q = str2sympy('sin(x)*sin(y)') # a scalar function in 2D
    """
    exp = sympify(expression)
    if isinstance(exp, (tuple, list)):
        return Matrix(exp)
    else:
        return exp


def sympy2exp(exp):
    """Convert a sympy expression to FEniCS expression.

    Args:

        exp (sympy.expr.Expr): Input expression

    Returns:

        str: FEniCS expression string

    Examples:

        >>> sympy2exp(str2sympy('sin(x)*sin(y)'))
        'sin(x[0])*sin(x[1])'
    """
    x, y, z = symbols('x[0] x[1] x[2]')

    def to_ccode(f):
        """Convert variable names."""
        f = f.subs('x', x).subs('y', y).subs('z', z)
        raw = printing.ccode(f)
        return raw.replace('M_PI', 'pi')

    if hasattr(exp, '__getitem__'):
        if exp.shape[0] == 1 or exp.shape[1] == 1:
            # Vector
            return tuple(map(to_ccode, exp))
        else:
            # Matrix
            return tuple([tuple(map(to_ccode, exp[i, :]))
                          for i in range(exp.shape[1])])
    else:
        # Scalar
        return to_ccode(exp)


def grad(u, dim=None):
    """Scalar, vector, or matrix gradient.

    If dim is not given, the dimension is inferred.

    Args:

        u (sympy.expr.Expr): function
        dim (int): dimension of the domain of the function

    Returns:

        sympy.expr.Expr: the gradient

    Examples:

        >>> v = str2sympy('sin(x)*sin(y)')
        >>> grad(v)
        Matrix([[sin(y)*cos(x), sin(x)*cos(y)]])
        >>> grad(v, dim=3)
        Matrix([[sin(y)*cos(x), sin(x)*cos(y), 0]])
    """
    if not dim:
        dim = infer_dim(u)
    # Transpose first if it is a row vector
    if u.is_Matrix and u.shape[0] != 1:
        u = u.transpose()
    # Take the gradient
    if dim == 1:
        return Matrix([u.diff('x')]).transpose()
    elif dim == 2:
        return Matrix([u.diff('x'), u.diff('y')]).transpose()
    elif dim == 3:
        return Matrix(
            [u.diff('x'), u.diff('y'), u.diff('z')]).transpose()


def curl(u):
    """Vector curl in 2D and 3D.

    Args:

        u (sympy.expr.Expr): function

    Returns:

        sympy.expr.Expr: the curl

    Examples:

        >>> u = str2sympy('sin(x)*sin(y)')
        >>> print(curl(u))
        Matrix([[sin(x)*cos(y)], [-sin(y)*cos(x)]])
        >>> v = str2sympy('(sin(y), sin(z), sin(x))')
        >>> print(curl(v))
        Matrix([[-cos(z)], [-cos(x)], [-cos(y)]])
    """
    if u.is_Matrix and len(u) == 3:
        # 3D vector curl
        return Matrix([u[2].diff('y') - u[1].diff('z'),
                       u[0].diff('z') - u[2].diff('x'),
                       u[1].diff('x') - u[0].diff('y')])
    else:
        # 2D rotated gradient
        return Matrix([u.diff('y'), -u.diff('x')])


def rot(u):
    """Vector rot in 2D. The result is a scalar function."""
    return u[1].diff('x') - u[0].diff('y')


def div(u):
    """Vector and matrix divergence.

    For matrices, the divergence is taken row-by-row.
    """
    def vec_div(w):
        """Vector divergence."""
        if w.shape[0] == 2:
            return w[0].diff('x') + w[1].diff('y')
        elif w.shape[0] == 3:
            return w[0].diff('x') + w[1].diff('y') + w[2].diff('z')

    if u.shape[1] == 1 and len(u.shape) == 2:
        # Column vector
        return vec_div(u)
    elif u.shape[0] == 1 and len(u.shape) == 2:
        # Row vector
        return vec_div(u.transpose())
    else:
        # Matrix
        result = []
        for i in range(u.shape[1]):
            result.append(vec_div(u.row(i).transpose()))
        return Matrix(result)


def sym(u):
    """Matrix symmetrization."""
    return (u + u.transpose()) / 2.0


def tr(u):
    """Matrix trace."""
    return u.trace()


def hess(u, dim=None):
    """The Hessian."""
    return grad(grad(u, dim), dim)


def star(u):
    """Unweighted Hodge star in Euclidean basis in 2D and 3D.

    In 2D, it rotates a vector counterclockwise by pi/2:

       [u0, u1] -> [-u1, u0]

    In 3D, it maps a vector to an antisymmetric matrix:

                       [0  -u2  u1]
       [u0, u1, u2] -> [ u2 0  -u0]
                       [-u1 u0  0 ]

    and it maps an antisymmetric matrix back to a vector reversing the above.
    """
    if len(u) == 2:
        # 2D
        return Matrix((-u[1], u[0]))
    elif len(u) == 3:
        # 3D
        if u.shape[0] * u.shape[1] == 3:
            # Vector
            return Matrix(((0, -u[2], u[1]),
                           (u[2], 0, -u[0]),
                           (-u[1], u[0], 0)))
        else:
            # Matrix
            if u.transpose() == -u:
                return Matrix((u[2, 1], u[0, 2], u[1, 0]))
            else:
                raise RuntimeError("Input matrix for Hodge star is not"
                                   "anti-symmetric.")


def eps(u):
    """Vector symmetric gradient."""
    return sym(grad(u).transpose())


def infer_dim(exp):
    """Infer the dimension of an expression."""
    atoms = exp.atoms()
    if sympify('z') in atoms:
        return 3
    elif sympify('y') in atoms:
        return 2
    else:
        return 1


if __name__ == "__main__":
    import doctest
    doctest.testmod()