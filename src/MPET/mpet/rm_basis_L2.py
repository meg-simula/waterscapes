from dolfin import *
import numpy as np

class MyExpression(Expression):
    def eval(self, value, x):
        value[0] = 1.0
    def value_shape(self):
        return (1,)

def rigid_motions(mesh):

    gdim = mesh.geometry().dim()
    x = SpatialCoordinate(mesh)
    c = np.array([assemble(xi*dx) for xi in x])
    volume = assemble(Constant(1)*dx(domain=mesh))
    c /= volume
    c_ = c
    c = Constant(c)

    if gdim == 1:       
        translations = [(MyExpression())]        
        return translations
    
    if gdim == 2:
        translations = [Constant((1./sqrt(volume), 0)),
                        Constant((0, 1./sqrt(volume)))]

        # The rotational energy
        r = assemble(inner(x-c, x-c)*dx)

        C0, C1 = c.values()
        rotations = [Expression(('-(x[1]-C1)/A', '(x[0]-C0)/A'), 
                                C0=C0, C1=C1, A=sqrt(r), degree=1)]
        
        return translations + rotations

    if gdim == 3:
        # Gram matrix of rotations
        R = np.zeros((3, 3))

        ei_vectors = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0,1))]
        for i, ei in enumerate(ei_vectors):
            R[i, i] = assemble(inner(cross(x-c, ei), cross(x-c, ei))*dx)
            for j, ej in enumerate(ei_vectors[i+1:], i+1):
                R[i, j] = assemble(inner(cross(x-c, ei), cross(x-c, ej))*dx)
                R[j, i] = R[i, j]

        # Eigenpairs
        eigw, eigv = np.linalg.eigh(R)
        if np.min(eigw) < 1E-8: warning('Small eigenvalues %g' % np.min(eigw))
        eigv = eigv.T
        # info('Eigs %r' % eigw)

        # Translations: ON basis of translation in direction of rot. axis
        # The axis of eigenvectors is ON but dont forget the volume
        translations = [Constant(v/sqrt(volume)) for v in eigv]

        # Rotations using the eigenpairs
        C0, C1, C2 = c_

        def rot_axis_v(pair):
            '''cross((x-c), v)/sqrt(w) as an expression'''
            v, w = pair
            return Expression(('((x[1]-C1)*v2-(x[2]-C2)*v1)/A',
                               '((x[2]-C2)*v0-(x[0]-C0)*v2)/A',
                               '((x[0]-C0)*v1-(x[1]-C1)*v0)/A'),
                               C0=C0, C1=C1, C2=C2, 
                               v0=v[0], v1=v[1], v2=v[2], A=sqrt(w),
                               degree=1)
        # Roations are discrebed as rot around v-axis centered in center of
        # gravity 
        rotations = list(map(rot_axis_v, zip(eigv, eigw)))
   
        return translations + rotations
