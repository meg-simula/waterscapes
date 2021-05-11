#from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

ai = [-0.0345, -0.0511, -0.0267, -0.0111, -0.0013, 0.0050,0.0027, 0.0061]
bi = [0.1009, 0.0284,-0.0160, -0.0070, -0.0174, -0.0041, -0.0041, 0.0005]
c0 = 102.3530
Hg2Pa = 133.3

t = np.arange(0.0, 1.0, 0.01)

pb = 0.01*(Hg2Pa * c0 * (1 + sum( [ai[i]*np.cos(2*np.pi*(i+1)*t) + bi[i]*np.sin(2*np.pi*(i+1)*t) for i in range(8)])) - 100*Hg2Pa)

ax = plt.gca()
ax.autoscale_view()

plt.plot(t, pb, linewidth=5)

plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Blood pressure (mean value has been subtracted)')
plt.grid(True)
png_folder = "../../png/"
plt.savefig(png_folder + "PBlood.png")
#plt.show()

plt.figure()
plt.plot(np.linspace(30,100,100), [0.05 * (x*x - 30.0*30.0)/(100.0*100.0 - 30.0*30.0) for x in np.linspace(30,100,100)],linewidth=5)

plt.xlabel('x [mm]')
plt.ylabel('V(x)')
plt.title('Volumetric fraction')
plt.grid(True)
png_folder = "../../png/"
plt.savefig(png_folder + "V(x).png")
#plt.show()

plt.figure()
plt.plot(np.linspace(0,1.2,100), [0.0041666*np.cos(2*np.pi/1.16030*t) for t in np.linspace(0,1.2,100)], linewidth=5)
plt.xlabel('Time [s]]')
plt.ylabel('g(t)')
plt.title('Volumetric source')
plt.grid(True)
png_folder = "../../png/"
plt.savefig(png_folder + "VolumetricSource.png")
#plt.show()


path = "/home/eleonora/PhD/MPET/src/sandbox/results/results_report/"
folders = listdir(path)
folders.remove('unused')
dt = 1.0/40.0
index = range(359, 400, 1)
plot_u = True
plot_p = True
time = np.linspace(9,10,41)


for folder in folders:

	#plt.close("all")
	XX = []
	UU = []
	PP = []
	TT = []
	
	p_30_file = open(path + folder + "/p_x_30_y_00.csv", "r")
	p_50_file = open(path + folder + "/p_x_50_y_00.csv", "r")
	p_70_file = open(path + folder + "/p_x_70_y_00.csv", "r")
	p_100_file = open(path + folder + "/p_x_100_y_00.csv", "r")

	u_30_file = open(path + folder + "/u_x_30_y_00.csv", "r")
	u_50_file = open(path + folder + "/u_x_50_y_00.csv", "r")
	u_70_file = open(path + folder + "/u_x_70_y_00.csv", "r")
	u_100_file = open(path + folder + "/u_x_100_y_00.csv", "r")

	p_x_30 = np.array([float(line.strip().split(",")[1]) for line in p_30_file.readlines()[1:]])
	p_x_50 = np.array([float(line.strip().split(",")[1]) for line in p_50_file.readlines()[1:]])
	p_x_70 = np.array([float(line.strip().split(",")[1]) for line in p_70_file.readlines()[1:]])
	p_x_100 = np.array([float(line.strip().split(",")[1]) for line in p_100_file.readlines()[1:]])

	ux_x_30 = np.array([float(line.strip().split(",")[1]) for line in u_30_file.readlines()[1:]])
	ux_x_50 = np.array([float(line.strip().split(",")[1]) for line in u_50_file.readlines()[1:]])
	ux_x_70 = np.array([float(line.strip().split(",")[1]) for line in u_70_file.readlines()[1:]])
	ux_x_100 = np.array([float(line.strip().split(",")[1]) for line in u_100_file.readlines()[1:]])

	plt.figure()
	plt.xlabel('time [s]')
	plt.ylabel('p [Pa]')
	plt.grid(True)
	plt.plot(np.linspace(8.0, 10.0, 81), p_x_30[319:], linewidth=1)
	plt.plot(np.linspace(8.0, 10.0, 81), p_x_50[319:], linewidth=1)
	plt.plot(np.linspace(8.0, 10.0, 81), p_x_70[319:], linewidth=1)
	plt.plot(np.linspace(8.0, 10.0, 81), p_x_100[319:], linewidth=1)
	print "max index p at x=30", p_x_30[319:].argmax()
	print "max index p at x=50", p_x_50[319:].argmax()
	print "max index p at x=70", p_x_70[319:].argmax()
	print "max index p at x=100", p_x_100[319:].argmax()

	print "min index p at x=30", p_x_30[319:].argmin()
	print "min index p at x=50", p_x_50[319:].argmin()
	print "min index p at x=70", p_x_70[319:].argmin()
	print "min index p at x=100", p_x_100[319:].argmin()
	#ax = plt.gca()
	#ax.autoscale_view()
	plt.savefig(png_folder + "p_x_30_" + folder + ".png")
	#plt.show()

	plt.figure()
	plt.xlabel('time [s]')
	plt.ylabel('ux [mm]')
	plt.grid(True)
	plt.plot(np.linspace(8.0, 10.0, 81), ux_x_30[319:], linewidth=1)
	plt.plot(np.linspace(8.0, 10.0, 81), ux_x_50[319:], linewidth=1)
	plt.plot(np.linspace(8.0, 10.0, 81), ux_x_70[319:], linewidth=1)
	plt.plot(np.linspace(8.0, 10.0, 81), ux_x_100[319:], linewidth=1)
	print "max index u at x=30", ux_x_30[319:].argmax()
	print "max index u at x=50", ux_x_50[319:].argmax()
	print "max index u at x=70", ux_x_70[319:].argmax()
	print "max index u at x=100", ux_x_100[319:].argmax()

	print "min index u at x=30", ux_x_30[319:].argmin()
	print "min index u at x=50", ux_x_50[319:].argmin()
	print "min index u at x=70", ux_x_70[319:].argmin()
	print "min index u at x=100", ux_x_100[319:].argmin()
	#ax = plt.gca()
	#ax.autoscale_view()
	plt.savefig(png_folder + "ux_x_30_" + folder + ".png")
	#plt.show()

	for i in index:
		x_y_0 = []
		ux_y_0 = []
		uy_y_0 = []
		u_y_0 = []


		u_file = open(path + folder + "/u_overaline_y_0." + str(i) + ".csv", "r")
		p_file = open(path + folder + "/p_overaline_y_0." + str(i) + ".csv", "r")
		

		lines_u = u_file.readlines()
		lines_p = p_file.readlines()

		x_y_0 = [float(line.strip().split(",")[8]) for line in lines_u[1:]] 
		ux_y_0 = [float(line.strip().split(",")[0]) for line in lines_u[1:]]
		uy_y_0 = [float(line.strip().split(",")[1]) for line in lines_u[1:]]
		u_y_0 = [np.sqrt(float(line.strip().split(",")[0])**2 + float(line.strip().split(",")[1])**2) for line in lines_u[1:]]   
		p_y_0 = np.array([float(line_p.strip().split(",")[0]) for line_p in lines_p[1:]]) 
		u_file.close()
		p_file.close()
        
		XX.append(x_y_0)
		UU.append(ux_y_0)
		PP.append(p_y_0)
		TT.append([time[i-index[0]]]*len(ux_y_0))

		plt.figure()
		plt.xlabel('x [mm]')
		plt.ylabel('ux [mm]')
		plt.grid(True)
		plt.plot(x_y_0, ux_y_0, linewidth=5)
		#ax = plt.gca()
		#ax.autoscale_view()
		plt.savefig(png_folder + "ux_t_" + str((i+1)*dt) + folder + ".png")

		#plt.show()

		plt.figure()
		plt.xlabel('x [mm]')
		plt.ylabel('uy [mm]')
		plt.grid(True)

		plt.plot(x_y_0, uy_y_0, linewidth=5)
		#ax = plt.gca()
		#ax.autoscale_view()

		plt.savefig(png_folder + "uy_t_" + str((i+1)*dt) + folder + ".png") 

		#plt.show()

		plt.figure()
		plt.xlabel('x [mm]')
		plt.ylabel('u [mm]')
		plt.grid(True)
		plt.plot(x_y_0, u_y_0, linewidth=5)
		#ax = plt.gca()
		#ax.autoscale_view()

		plt.savefig(png_folder + "u_t_" + str((i+1)*dt) + folder + ".png")

		#plt.show()

		plt.figure()
		plt.xlabel('x [mm]')
		plt.ylabel('p [Pa]')
		plt.grid(True)
		plt.plot(x_y_0, p_y_0, linewidth=5)
		#ax = plt.gca()
		#ax.autoscale_view()
		if p_y_0.min() == p_y_0.max():
			plt.savefig(png_folder + "p_t_" + str((i+1)*dt) + folder + ".png")
		else:			
			plt.yticks(np.arange(p_y_0.min()-0.5, p_y_0.max() + 0.5,  (p_y_0.max() + 0.5 - p_y_0.min() + 0.5)/5.0),\
			           np.arange(p_y_0.min()-0.5, p_y_0.max() + 0.5, (p_y_0.max() + 0.5 - p_y_0.min() + 0.5)/5.0))
	#		plt.yticks(np.arange(1.1*p_y_0.min(), 0.9*p_y_0.max(), (1.1*p_y_0.max() - 0.9*p_y_0.min())/10.0),\
	#		           np.arange(1.1*p_y_0.min(), 0.9*p_y_0.max(), (1.1*p_y_0.max() - 0.9*p_y_0.min())/10.0))

			#plt.show()
			plt.savefig(png_folder + "p_t_" + str((i+1)*dt) + folder + ".png")

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(TT, XX, UU, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	ax.view_init(elev=29, azim=-66)
	ax.set_xlabel("Time [s]")
	ax.set_ylabel("x [mm]")
	ax.set_zlabel("ux [mm]")
	plt.savefig(png_folder + "ux_alltime_" + folder + ".png")

	#plt.show()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(TT, XX, PP, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	ax.view_init(elev=29, azim=-66)
	ax.set_xlabel("Time [s]")
	ax.set_ylabel("x [mm]")
	ax.set_zlabel("p [Pa]")
	plt.savefig(png_folder + "p_alltime_" + folder + ".png")
