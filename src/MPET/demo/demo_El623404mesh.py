__author__ = "Eleonora Piersanti (eleonora@simula.no) and Marie E. Rognes (meg@simula.no), 2017"

import math
# import pytest
from mpet import *
from matplotlib import pylab
from datetime import datetime
import os
import csv
import os.path
# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
set_log_level(15)

def solid_pressure(u, E, nu):
    "Define the standard linear elastic constitutive equation."
    d = u.geometric_dimension()
    I = Identity(d)
    mu = E/(2.0*((1.0 + nu)))
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    s = 2*mu*sym(grad(u)) + lmbda*div(u)*I
    ps = -1.0/d*sum([s[(i,i)] for i in range(d)])
    return ps

def biot_El623404():

    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    theta = 1.0
    dt = 0.01
    T = dt
    # Define material parameters in MPET equations
    A = 1
    c = 1.0e-4
    alpha = (1.0,)
    K = (1.0e-5,)
    S = ((0.0,),)
    E = 5000 # Pa
    nu = 0.479
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Setting up MPET problem")

    # Read mesh in xml and save it in hdf5 format if it does not exists 
    hdf5_mesh_file = "../../../meshes/El623404/El623404.h5"
    if not(os.path.isfile(hdf5_mesh_file)):
        mesh_xml = Mesh("../../../meshes/El623404/El623404.xml")
        subdomains_xml = MeshFunction("size_t", mesh_xml, "../../../meshes/El623404/El623404_subdomains.xml")
        mesh_file = HDF5File(mpi_comm_world(), "../../../meshes/El623404/El623404.h5", 'w')
        mesh_file.write(mesh_xml, '/mesh')
        mesh_file.write(subdomains_xml, "/subdomains")
        mesh_file.close()

    mesh = Mesh()
    mesh_file = HDF5File(mpi_comm_world(), '../../../meshes/El623404/El623404.h5', 'r')
    mesh_file.read(mesh, '/mesh', False)
    sub_domains = MeshFunction("size_t", mesh)
    mesh_file.read(sub_domains, "/subdomains")
    MeshPartitioning.build_distributed_mesh(mesh)

    sub_cells = CellFunction("size_t", mesh, 0)

    exit()
    # mesh = UnitCubeMesh(50,50,50)

    # sub_domains = MeshFunction("size_t", mesh, 3)
    # sub_cells = CellFunction("size_t", mesh, 0)

    print(len(sub_cells))
    print(len(sub_domains))
    for i in range(len(sub_cells)):
        sub_cells[i] = sub_domains[i]

    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    
    all_boundary = CompiledSubDomain('on_boundary')

    problem.displacement_nullspace = False
    all_boundary.mark(problem.momentum_boundary_markers, 0)
    for i in range(A):
        all_boundary.mark(problem.continuity_boundary_markers[i], 0)

    DG = FunctionSpace(mesh, "DG", 0)

    f = HDF5File(mpi_comm_world(),'p_bar.h5', 'r')
    pressure = Function(DG)
    f.read(pressure, "/initial")

    # for i in range(len(sub_cells)):
    #     print "i = ", i
    #     if sub_cells[i] == 4 or sub_cells[i] == 6:
    #         pressure.vector()[i] = 133.322
 
    # problem.p_bar = [pressure,]
    problem.u_bar = Constant((0.0,0.0,0.0))
    problem.f = Constant((0.0,0.0,0.0))
    problem.s = Constant((0.0,0.0,0.0))
    # Set-up solver
    params = dict(dt=dt, theta=theta, T=T, stabilization=False)
    solver = MPETSolver(problem, params)

    # Using zero initial conditions by default
    
    # Solve
    # points = [Point(30.0,0.0), Point(50.0,0.0), Point(70.0,0.0), Point(100.0,0.0)]
    solutions = solver.solve()
    up = solver.up_
    # u0_values = [[up(point)[0] for point in points],]
    # p_values = [[up(point)[2] for point in points],]

    # PS = FunctionSpace(mesh, "CG", 1)
    # ps_0 = solid_pressure(up.sub(0), E, nu)
    # ps_0 = project(ps_0, PS)

    today = datetime.now()
    foldername = "results/"+today.strftime('%Y%m%d_%H%M%S')
    # os.makedirs(foldername)
    Fileu = File(foldername + "/u_robin.pvd")
    Filep = File(foldername + "/p_robin.pvd")

    # params_file = csv.writer(open(foldername+"/params.csv", "w"))
    # for key, val in problem.params.items():
    #     params_file.writerow([key, val])

    # params_file.writerow(["IB_M", IB_M])
    # params_file.writerow(["OB_M", OB_M])
    # params_file.writerow(["IB_C", IB_C])
    # params_file.writerow(["OB_C", OB_C])    
    # ps_values = [[ps_0(point) for point in points],]
    # times = [0.0,]
    for (up, t) in solutions:
        info("t = %g" % t)
        Fileu << up.sub(0)
        Filep << up.sub(1)         
        # plot(up.sub(0), key="u")
        # plot(up.sub(1), key="p0")
        # times += [t]
        # u0_values.append([up(point)[0] for point in points])
        # p_values.append([up(point)[2] for point in points])

        # ps = solid_pressure(up.sub(0), E, nu)
        # ps = project(ps, PS)
        # ps_values.append([ps(point) for point in points])
    #     plot(ps, key="ps")

    # interactive()

    # a = zip(*u0_values)
    # b = zip(*p_values)
    # d = zip(*ps_values)

    # pylab.figure()
    # for (k, i) in enumerate(a):
    #     pylab.plot(times, i, "-*", label ="x_%d" %k)
    # pylab.grid(True)
    # pylab.xlabel("time")
    # pylab.ylabel("u0")
    # pylab.legend()
    # pylab.savefig(foldername + "/u.png")

    # pylab.figure()
    # for (k, i) in enumerate(b):
    #     pylab.plot(times, i, "-*", label ="x_%d" %k)
    # pylab.grid(True)
    # pylab.xlabel("time")
    # pylab.ylabel("p")
    # pylab.legend()
    # pylab.savefig(foldername + "/p.png")

    # pylab.figure()
    # for (k, i) in enumerate(d):
    #     pylab.plot(times, i, "-*", label ="x_%d" %k)
    # pylab.grid(True)
    # pylab.xlabel("time")
    # pylab.ylabel("ps")
    # pylab.legend()
    # pylab.savefig(foldername + "/p_solid.png")
    # pylab.show()

def create_markers():

    mesh = Mesh("../../../meshes/El623404/El623404.xml")
    sub_domains = MeshFunction("size_t", mesh, "../../../meshes/El623404/El623404_subdomains.xml")
    facets_momentum = FacetFunction('size_t', mesh, 7101982)
    facets_continuity = FacetFunction('size_t', mesh, 7101982)
    sub_cells = CellFunction("size_t", mesh,0)

    for i in range(len(sub_cells)):
        sub_cells[i] = sub_domains[i]
    
    File("cellfunction.xml")<<sub_cells
    for cell in SubsetIterator(sub_cells, 4):
        for facet in facets(cell): 
            facets_momentum[facet] = 0

    for cell in SubsetIterator(sub_cells, 6):
        for facet in facets(cell): 
            facets_momentum[facet] = 0

    for cell in SubsetIterator(sub_cells, 4):
        for facet in facets(cell): 
            facets_continuity[facet] = 0

    for cell in SubsetIterator(sub_cells, 6):
        for facet in facets(cell): 
            facets_continuity[facet] = 0

    DG = FunctionSpace(mesh, "DG", 0)
    pressure = Function(DG)
    for i in range(len(sub_cells)):
        if sub_cells[i] == 4 or sub_cells[i] == 6:
            pressure.vector()[i] = 133.322
    f = HDF5File(mpi_comm_world(),'p_bar.h5', 'w')
    f.write(pressure, '/initial')

    f = HDF5File(mpi_comm_world(),'facets_momentum.h5', 'w')
    f.write(facets_momentum, '/facets')
    f = HDF5File(mpi_comm_world(),'facets_continuity.h5', 'w')
    f.write(facets_continuity, '/facets')

    # f = HDF5File(mpi_comm_world(),'p_bar.h5', 'w')
    # f.write(pressure, '/initial')

    File("facets_momentum.xml")<<facets_momentum
    File("facets_continuity.xml")<<facets_continuity
    File("facets_momentum.pvd")<<facets_momentum
    File("facets_continuity.pvd")<<facets_continuity
    


    
if __name__ == "__main__":

    # create_markers()
    biot_El623404()

# 