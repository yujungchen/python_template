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
	

array1 = [ ]
array2 = [ ]

#BSSRDF parameter
sigma_a = [2.19, 2.62, 3.00]
sigma_s = [0.0021, 0.0041, 0.0071]
sigma_t = np.add(sigma_a, sigma_s)
print ('Sigma_s (', sigma_s[0], sigma_s[1], sigma_s[2], ')')
print ('Sigma_a (', sigma_a[0], sigma_a[1], sigma_a[2], ')')
print ('Sigma_t (', "%.5f" % sigma_t[0], "%.5f" % sigma_t[1], "%.5f" % sigma_t[2], ')')


total = len(sys.argv)
cmdargs = str(sys.argv)

# NUmber of samples
sampleNum = int(sys.argv[1])


for i in range(sampleNum):
	#Random sampling
	x, y = RandomSampling()
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