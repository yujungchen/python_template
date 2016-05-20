#!/usr/bin/python

import numpy as np
import random as rnd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

#Generate gaussian random number
def GaussianSampling():
	u1 = rnd.random()
	u2 = rnd.random()
	r = np.sqrt(-2.0 * np.log(u1))
	theta = 2.0 * np.pi * u2
	nx = r * np.cos(theta)
	return nx

#2D gaussian random number
def Gaussian2DSampling(mu_x, sigma_x, mu_y, sigma_y, rho):
	sample_x = GaussianSampling()
	sample_y = GaussianSampling()
	x = sigma_x * sample_x + mu_x	
	y = sigma_y * (rho * sample_x + np.sqrt(1.0 - rho * rho) * sample_y) + mu_y
	return (x, y)

def Gaussian3DSampling(mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z, alpha, beta, gamma):
	sample_x = GaussianSampling()
	sample_y = GaussianSampling()
	sample_z = GaussianSampling()
	x = sigma_x * sample_x + mu_x	
	y = sigma_y * (alpha * sample_x + np.sqrt(1.0 - alpha * alpha) * sample_y) + mu_y
	temp = gamma - alpha * beta
	z = sigma_z * (beta * sample_x + 
				   np.sqrt(1.0 - alpha * alpha) * temp * sample_y + 
				   np.sqrt( (1.0 - beta * beta - temp * temp / (1.0 - alpha * alpha) ) ) * sample_z ) + mu_z
	return (x, y, z)

total = len(sys.argv)
cmdargs = str(sys.argv)

# NUmber of samples
sampleNum = int(sys.argv[1])

array1 = [ ]
array2 = [ ]
array3 = [ ]

for i in range(sampleNum):
	rx, ry, rz = Gaussian3DSampling(0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0)
	array1.append(rx)
	array2.append(ry)
	array3.append(rz)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for c, m in [('r', 'o')]:
	ax.scatter(array1, array2, array3, c = c, marker = m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()