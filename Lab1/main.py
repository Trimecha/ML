import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from lab_ML_supervised import \
    rand_gauss, rand_bi_gauss, rand_clown, rand_checkers, grid_2d, plot_2d, \
    frontiere, mse_loss, gradient, plot_gradient, poly2, collist, \
    symlist, gr_mse_loss, hinge_loss, gr_hinge_loss

from Data_generation import *
from Logistic_regression import *

########################################## marianne.clausel@imag.fr
### Main script

def main() :
        #plot_random(100)
        dataX,dataY = save_random(100,100)
        print("size dataX = ",dataX.size," ; size dataY = ",dataY.size)
        #plot_2d(dataX,dataY)
        #plt.show()
        logisticRegression(dataX,dataY)
        return 0



if __name__ == "__main__":
	main()
