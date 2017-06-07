from dolfinimport import errornorm
from dolfinimport import VectorFunctionSpace, FunctionSpace, interpolate

def compute_error(u0, p0, uex, pex, mesh):
    # V_ex = VectorFunctionSpace(mesh, "CG", 5)
    # W_ex = FunctionSpace(mesh, "CG", 5)
    erru = errornorm(uex, u0, norm_type='H1', degree_rise=3, mesh=mesh) #do not interpolate and degree_rise = 3 is enough
    errpL2 = [errornorm(pex[i], p0[i], norm_type='L2', degree_rise=3, mesh=mesh)
            for i in range(len(p0))]
    errpH1 = [errornorm(pex[i], p0[i], norm_type='H1', degree_rise=3, mesh=mesh)
            for i in range(len(p0))]
    return erru, errpL2, errpH1


def make_latex_table(x, filename):
    
    filename.write("$")
    filename.write(x[0])
    filename.write("$")
    
    for i in range(1,len(x)):
        filename.write(" & ")
        filename.write("$")
        filename.write(x[i])
        filename.write("$")
    filename.write(" \\")
    filename.write("\\")
    filename.write("\n")
    
class SaveError():

    
    def __init__ (self, NN, dts, AA, param_str, folder):

        self.param_str = param_str
        cols = len(NN)+1
        rows = len(dts)+1
        self.Erroru = [[0] * cols for i in range(rows)]
        self.ErrorpL2 = [ [[0] * cols for i in range(rows)] for i in range(AA) ]
        self.ErrorpH1 = [ [[0] * cols for i in range(rows)] for i in range(AA) ]

        self.Erroru[0][0] = "\t\t\t\t\t"
        self.Erroru[0][1:cols] = ["N = " + str(NN[i]) + "\t\t\t" for i in range(0, cols-1)]
       
        for i in range(1,rows):
            self.Erroru[i][0] = "dt = T/" + str(NN[i-1]) + "\t\t"


        for j in range(AA):
            self.ErrorpL2[j][0][0] = "\t\t\t\t\t"
            self.ErrorpL2[j][0][1:cols] = ["N = " + str(NN[i]) + "\t\t\t" for i in range(0, cols-1)]
            self.ErrorpH1[j][0][0] = "\t\t\t\t\t"
            self.ErrorpH1[j][0][1:cols] = ["N = " + str(NN[i]) + "\t\t\t" for i in range(0, cols-1)]
            for k in range(1, rows):
                self.ErrorpL2[j][k][0] = "dt = T/" + str(NN[k-1]) + "\t\t\t"
                self.ErrorpH1[j][k][0] = "dt = T/" + str(NN[k-1]) + "\t\t\t"


        self.erroru = open(folder + '/error_u' + param_str, 'w')
        self.errorpL2 = [open(folder + '/error_p' + str(i) + 'L2' + param_str , 'w') for i in range(AA)]
        self.errorpH1 = [open(folder + '/error_p' + str(i) + 'H1' + param_str , 'w') for i in range(AA)]


    def store_error(self, erru, errpL2, errpH1, cc, rr):

        self.Erroru[rr][cc] = erru

        for i in range(len(errpL2)):
            self.ErrorpL2[i][rr][cc] = errpL2[i]
            
        for i in range(len(errpH1)):
            self.ErrorpH1[i][rr][cc] = errpH1[i]

    def save_file(self, AA):

        for row in self.Erroru:
            for column in row:
                self.erroru.write("$")
                self.erroru.write(str(column))
                self.erroru.write("$\t\t")
                self.erroru.write("&")
            self.erroru.write('\\\\')
            self.erroru.write('\n')

        for i in range(AA):

            for row in self.ErrorpL2[i][:][:]:
                for column in row:
                    self.errorpL2[i].write("$")
                    self.errorpL2[i].write(str(column) + "\t\t")
                    self.errorpL2[i].write("$\t\t")
                    self.errorpL2[i].write("&")
                self.errorpL2[i].write('\\\\')
                self.errorpL2[i].write('\n')

        for i in range(AA):

            for row in self.ErrorpH1[i][:][:]:
                for column in row:
                    self.errorpH1[i].write("$")
                    self.errorpH1[i].write(str(column) + "\t\t")
                    self.errorpH1[i].write("$\t\t")
                    self.errorpH1[i].write("&")
                self.errorpH1[i].write('\\\\')
                self.errorpH1[i].write('\n')
                
        