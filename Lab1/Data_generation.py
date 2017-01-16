import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from lab_ML_supervised import \
    rand_gauss, rand_bi_gauss, rand_clown, rand_checkers, grid_2d, plot_2d, \
    frontiere, mse_loss, gradient, plot_gradient, poly2, collist, \
    symlist, gr_mse_loss, hinge_loss, gr_hinge_loss


def plot_random(n) :
        mu=[1,1]
        sigma=[1,1]
        data = rand_gauss(n,mu,sigma)
        plt.plot(data)
	#plt.show()

        n1=20
        n2=20
        mu1=[1,1]
        mu2=[-1,-1]
        sigma1=[0.9,0.9]
        sigma2=[0.9,0.9]
        data1=rand_bi_gauss(n1,n2,mu1,mu2,sigma1,sigma2)
        plt.plot(data1)
	#plt.show()

        std1=1
        std2=5
        n1=50
        n2=50
        data2=rand_clown(n1,n2,std1,std2)
        plt.plot(data2)
	#plt.show()

        std=0.1
        data3=rand_checkers(n1,n2,std)
        plt.plot(data3)
	#plt.show()
        dataX=data1[:,:2] # we only take the first two features.
        print(dataX.size)
        dataY=data1[:,2]

def save_random(n1,n2):
        mu1=[1,1]
        mu2=[-1,-1]
        sigma1=[0.9,0.9]
        sigma2=[0.9,0.9]
        data1=rand_bi_gauss(n1,n2,mu1,mu2,sigma1,sigma2)
        dataX=data1[:,:2]
        dataY=data1[:,2]
        #print(data1)
        return dataX,dataY
