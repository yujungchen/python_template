#!/usr/bin/python

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

array1 = [ ]
array2 = [ ]

for i in range(5):
	array1.append(rnd.random())
	array2.append(rnd.random())

for i in range(5):
	print (array1[i], array2[i])

t = np.arange(0., 5., 0.2)
plt.plot(array1, array2, 'g^')
plt.show()