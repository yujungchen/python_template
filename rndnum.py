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
	DiscR = -1.0 * np.log(1.0 - Seed) / _sigma_tr;
	TempDisc = 0.5 * np.sqrt(rnd.random())
	angle = rnd.random()
	x_angle = 2.0 * np.pi * angle
	y_angle = 2.0 * np.pi * angle
	x = DiscR * np.cos(x_angle);
	y = DiscR * np.sin(y_angle);
	return (x, y)

array1 = [ ]
array2 = [ ]

#BSSRDF parameter
one = [1.0, 1.0, 1.0]
sigma_a = [2.19, 2.62, 3.00]
sigma_s = [0.0021, 0.0041, 0.0071]
sigma_t = np.add(sigma_a, sigma_s)
sigma_tr = 3.0 * np.multiply(sigma_a, sigma_t)
sigma_tr = np.sqrt(sigma_tr)
alpha_prime = np.divide(sigma_s, sigma_t)
zr = np.divide(one, sigma_t)
eta = 1.3
Fdr = -1.4399 / (eta * eta) + 0.7099 / eta + 0.6681 + 0.0636 * eta;
A = (1.0 + Fdr) / (1.0 - Fdr)
scale =  (1.0 + 4.0 / 3.0 * A)
zv = zr * scale

print ('\nBSSRDF Parameters')
print ('Sigma_s (', sigma_s[0], sigma_s[1], sigma_s[2], ')')
print ('Sigma_a (', sigma_a[0], sigma_a[1], sigma_a[2], ')')
print ('Sigma_t (', "%.5f" % sigma_t[0], "%.5f" % sigma_t[1], "%.5f" % sigma_t[2], ')')
print ('Sigma_tr (', "%.5f" % sigma_tr[0], "%.5f" % sigma_tr[1], "%.5f" % sigma_tr[2], ')')
print ('Alpha Prime (', "%.5f" % alpha_prime[0], "%.5f" % alpha_prime[1], "%.5f" % alpha_prime[2], ')')
print ('zr (', "%.5f" % zr[0], "%.5f" % zr[1], "%.5f" % zr[2], ')')
print ('zv (', "%.5f" % zv[0], "%.5f" % zv[1], "%.5f" % zv[2], ')')
#print ('scale', "%.5f" % scale)
print ('Fdr', "%.5f" % Fdr)
print ('A', "%.5f" % A)


total = len(sys.argv)
cmdargs = str(sys.argv)

# NUmber of samples
sampleNum = int(sys.argv[1])


for i in range(sampleNum):
	#Uniform Random sampling
	#x, y = RandomSampling()
	x, y = ExponentialSampling(sigma_tr[0])
	array1.append(x)
	array2.append(y)

#for i in range(sampleNum):
#	print (array1[i], array2[i])

#Figure configuration
plt.figure(figsize = (10, 10))
plt.plot(array1, array2, 'r.')
plt.title('Distribution Plot')
plt.grid(True)
plt.axis([-0.6, 0.6, -0.6, 0.6])
plt.axes().set_aspect('equal', 'datalim')
plt.show()