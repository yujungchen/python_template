#!/usr/bin/python

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import sys

#Test 2D radnom number generation
def RandomSampling():
	x = rnd.random() - 0.5
	y = rnd.random() - 0.5
	return (x, y)

#Test 2D exponential random number generation	
def ExponentialSampling(_sigma_tr):
	Seed = rnd.random()
	DiscR = -1.0 * np.log(1.0 - Seed) / _sigma_tr
	TempDisc = 0.5 * np.sqrt(rnd.random())
	rndangle = rnd.random()
	angle = 2.0 * np.pi * rndangle
	x = DiscR * np.cos(angle)
	y = DiscR * np.sin(angle)
	return (x, y)

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


array1 = [ ]
array2 = [ ]


total = len(sys.argv)
cmdargs = str(sys.argv)

# NUmber of samples
sampleNum = int(sys.argv[1])


for i in range(sampleNum):
	#Uniform Random sampling
	#x, y = RandomSampling()
	rx, ry = Gaussian2DSampling(0.0, 2.0, 0.0, 1.0, -0.75)
	array1.append(rx)
	array2.append(ry)

mean_array1 = np.mean(array1)
mean_array2 = np.mean(array2)
print ('Sample Mean (', "%.5f" %mean_array1, "%.5f" %mean_array2, ')')

#for i in range(sampleNum):
#	print (array1[i], array2[i])

#Figure configuration
plt.figure(figsize = (10, 10))
plt.plot(array1, array2, 'r.')
plt.title('Distribution Plot')
plt.grid(True)
plt.axis([-10.0, 10.0, -10.0, 10.0])
plt.axes().set_aspect('equal', 'datalim')
plt.show()