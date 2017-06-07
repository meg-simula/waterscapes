import matplotlib.pyplot as plt
import sys
import numpy as np
from tabulate import tabulate

folder1 = "2D_1Net_PulsatilePressure_FixedSkull_NoFluxOnVentricles_nullspace_False_E_5.000e+02_nu_4.9999e-01_K_1.500e-05_Q2.222e+09_Donut_coarse"
folder2 = "2D_1Net_PulsatilePressure_FixedSkull_NoFluxOnVentricles_nullspace_False_E_5.000e+03_nu_4.9999e-01_K_1.50000e-05_Q2.222e+09_Donut_coarse"
folder3 = "2D_1Net_PulsatilePressure_FixedBrain_nullspace_False_E_5.000e+03_nu_4.99990e-01_K_1.50000e-05_Q2.222e+09_Donut_coarse"
folder4 = "2D_1Net_PulsatilePressure_FixedBrain_nullspace_False_E_5.000e+02_nu_3.50000e-01_K_1.50000e-05_Q2.222e+09_Donut_coarse"
folder5 = "2D_1Net_PulsatilePressure_FixedBrain_nullspace_False_E_5.000e+02_nu_4.99990e-01_K_1.50000e-05_Q2.222e+09_Donut_coarse"
folder6 = "2D_1Net_PulsatilePressure_FixedBrain_nullspace_False_E_5.000e+01_nu_4.99990e-01_K_1.50000e-05_Q2.222e+09_Donut_coarse"
# folder = [folder1, folder2, folder3, folder4, folder5, folder6]

# folder = ["2D_1Net_PulsatilePressure_FixedBrain_nullspace_False_E_5.000e+02_nu_4.99990e-01_K_1.50000e-05_Q2.222e+09_Donut_coarse_refined"]
folder = ["2D_1Net_PulsatilePressure_FixedBrain_Acceleration_nullspace_False_E_5.000e+02_nu_3.50000e-01_K_1.50000e-05_Q2.222e+09_Donut_coarse_refined"]

dt = 1.0/40.0
# foldername = sys.argv[1]
for foldername in folder:
    p30 = []
    p50 = []
    p70 = []
    
    for i in range(400):
        namefile = foldername+"/p_overaline." + str(i) + ".csv"
        f = open(namefile, "r")
        next(f)
        for line in f:
            a = line.strip().split(",")
            [p,x] = [a[0],a[3]]
    
            if float(x) == 30.0:
                p30 += [float(p)]            
            if float(x) == 50.0:
                p50 += [float(p)]
            elif float(x) == 70.0:
                p70 += [float(p)]
    
    # print p70
    print foldername
    table = [ [np.max(p30), 30, "%04.06f" %(dt*(np.argmax(p30)+1))] , ["%04.06f" %np.max(p50), 50, "%04.06f" %(dt*(1+np.argmax(p50)))], ["%04.06f" %np.max(p70), 70, "%04.06f" %(dt*(1+np.argmax(p70)))]]
    # table = [[np.max(p30), 30, dt*np.argmax(p30)]]
    headers = ["maxp", "position", "time"]
    # print table
    # print "maximum value of the pressure at x = 30 is %04.06f" %np.max(p30), "at t = %04.06f" %(dt*np.argmax(p30))
    # print "maximum value of the pressure at x = 50 is %04.06f" %np.max(p50), "at t = %04.06f" %(dt*np.argmax(p50))
    # print "maximum value of the pressure at x = 70 is %04.06f" %np.max(p70), "at t = %04.06f" %(dt*np.argmax(p70))

    print tabulate(table, headers, tablefmt="latex")
    p30_50 = [float(a_i) - float(b_i) for a_i, b_i in zip(p30, p50)]
    p50_70 = [float(a_i) - float(b_i) for a_i, b_i in zip(p50, p70)]
    p30_70 = [float(a_i) - float(b_i) for a_i, b_i in zip(p30, p70)]
    
    # p50_70 = 
    fig, ax = plt.subplots()
    ax.plot(p30, label="x = 30")
    ax.plot(p50, label="x = 50")
    ax.plot(p70, label="x = 70")
    legend = ax.legend(loc='best', shadow=True)
    
    plt.savefig(foldername+"/" + "pressure.png", format='png', dpi=1200)
    # plt.show()
    # plt.close(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(p30_50, label="x = 30 - x=50")
    plt.savefig(foldername+"/" + "p30_50.png", format='png', dpi=1200)
    # plt.show()
    # plt.close(fig2)
    
    fig3, ax3 = plt.subplots()
    ax3.plot(p50_70, label="x = 50 - x=70")
    plt.savefig(foldername+"/" + "p50_70.png", format='png', dpi=1200)
    # plt.show()
    # plt.close(fig3)
    
    
    fig4, ax4 = plt.subplots()
    ax4.plot(p30_70, label="x = 30 - x=50")
    plt.savefig(foldername+"/" + "p30_70.png", format='png', dpi=1200)
    # plt.show()
    # plt.close(fig4)
