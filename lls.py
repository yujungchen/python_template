#!/usr/bin/python

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import sys

# Test 2D radnom number generation
def RandomSampling():
	x = rnd.random() - 0.5
	y = rnd.random() - 0.5
	return (x, y)

# Test 2D exponential random number generation	
def ExponentialSampling(_sigma_tr):
	Seed = rnd.random()
	DiscR = -1.0 * np.log(1.0 - Seed) / _sigma_tr
	TempDisc = 0.5 * np.sqrt(rnd.random())
	rndangle = rnd.random()
	angle = 2.0 * np.pi * rndangle
	x = DiscR * np.cos(angle)
	y = DiscR * np.sin(angle)
	return (x, y)

# Generate gaussian random number
def GaussianSampling():
	u1 = rnd.random()
	u2 = rnd.random()
	r = np.sqrt(-2.0 * np.log(u1))
	theta = 2.0 * np.pi * u2
	nx = r * np.cos(theta)
	return nx

# 2D gaussian random number
def Gaussian2DSampling(mu_x, sigma_x, mu_y, sigma_y, rho):
	sample_x = GaussianSampling()
	sample_y = GaussianSampling()
	x = sigma_x * sample_x + mu_x	
	y = sigma_y * (rho * sample_x + np.sqrt(1.0 - rho * rho) * sample_y) + mu_y
	return (x, y)

# Compute cost
def ComputeCost(x, y, m, b):
	cost = 0
	num = len(x)

	for idx in range(num):
		#print(idx, '%.3f' % x[idx], '%.3f' % y[idx])
		ErrorTerm = y[idx] - m * x[idx] - b
		cost = cost + ErrorTerm

	cost = cost / num
	return cost

# Main function
if __name__ == '__main__':
	data_x = [ ]
	data_y = [ ]
	cost_log = [ ]
	cmdargs = str(sys.argv)

	# Number of samples
	sampleNum = int(sys.argv[1])
	iteration_cnt = int(sys.argv[2])
	epsilon = (1.0 / sampleNum)  / 100000.0
	print "Stopping criteria:", epsilon

	# Generate pesudo data
	for i in range(sampleNum):
		#Uniform Random sampling
		rx, ry = Gaussian2DSampling(0.0, 1.0, 1.0, 2.0, 0.75)
		data_x.append(rx)
		data_y.append(ry)

	# Initialize parameters
	a = 0.0
	b = 0.0
	alpha = 0.0001	#Learning rate
	x = np.arange(-100, 100)

	cost = ComputeCost(data_x, data_y, a, b)
	cost_log.append(cost)

	iteration = 0
	TotalSample = sampleNum

	plt.figure(1, figsize = (10, 10))
	plt.plot(data_x, data_y, 'r.')

	for iteration in range(iteration_cnt):
	#	print(iteration)
		GradientSum_a = 0.0
		GradientSum_b = 0.0
		
		for datanum in range(TotalSample):	
			#Compute gradient
			G = 0.0
			if data_x[datanum] < 10.0 and data_x[datanum] > -10.0 and data_y[datanum] < 10.0 and data_y[datanum] > -10.0 :
				G = (data_y[datanum] - (a * data_x[datanum] + b))
				GradientSum_a = GradientSum_a + G * data_x[datanum]
				GradientSum_b = GradientSum_b + G
		
		a = a + alpha * GradientSum_a
		b = b + alpha * GradientSum_b
		cost = ComputeCost(data_x, data_y, a, b)
		cost_log.append(cost)
		# Stop gradient descent at some epsilon
		if cost < epsilon : 
			print "Stop gradient descent at iteration", iteration
			break


	# Fitted parameter
	y = a * x + b
	m, b = np.polyfit(x, y, 1)
	plt.plot(x, m * x + b, '-')

	# Plot data and fitted line
	plt.title('Simple Linear Least Square')
	plt.grid(True)
	plt.axis([-10.0, 10.0, -10.0, 10.0])
	plt.axes().set_aspect('equal', 'datalim')

	# Plot cost trend
	plt.figure(2)
	plt.title('Cost')
	plt.plot(cost_log, '-')
	plt.grid(True)
	plt.show()