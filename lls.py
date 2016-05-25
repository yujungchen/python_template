#!/usr/bin/python

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import sys
import time

class Sample :
	def __init__(self, _x, _y) :
		self.x = _x
		self.y = _y

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
def ComputeCost(Sample, m, b):
	cost = 0
	num = len(Sample)

	for idx in range(num):
		#print(idx, '%.3f' % x[idx], '%.3f' % y[idx])
		ErrorTerm = Sample[idx].y - m * Sample[idx].x - b
		cost = cost + ErrorTerm * ErrorTerm

	cost = cost / num
	return cost


# Main function
if __name__ == '__main__':
	data_x = [ ]
	data_y = [ ]
	SynSample = [ ]
	cost_log = [ ]
	cmdargs = str(sys.argv)

	# Number of samples
	sampleNum = int(sys.argv[1])
	iteration_cnt = int(sys.argv[2])
	mode = int(sys.argv[3])
	#epsilon = (1.0 / sampleNum)  / 100000.0
	#print ("Stopping criteria:", epsilon)
	prev_cost = 0

	# Generate pesudo data
	for i in range(sampleNum):
		#Uniform Random sampling
		rx, ry = Gaussian2DSampling(0.0, 1.0, 1.0, 2.0, 0.75)
		SynSample.append(Sample(rx, ry))
		data_x.append(rx)
		data_y.append(ry)


	# Initialize parameters
	a = 0.0
	b = 0.0
	alpha = 0.0001	#Learning rate
	x = np.arange(-100, 100)
	cost = ComputeCost(SynSample, a, b) 
	cost_log.append(cost)

	iteration = 0
	
	plt.figure(1, figsize = (10, 10))
	plt.plot(data_x, data_y, 'r.')

	if mode == 1 :
		print ("Stochastic gradient descent")
		t0 = time.clock()
		for iteration in range(iteration_cnt) :
			#np.random.shuffle(SynSample)
			for datanum in range(sampleNum):
				G = 0.0
				if SynSample[datanum].x < 10.0 and SynSample[datanum].x > -10.0 and SynSample[datanum].y < 10.0 and SynSample[datanum].y > -10.0 :
					G = (SynSample[datanum].y - (a * SynSample[datanum].x + b))
			
				a = a + alpha *	G * SynSample[datanum].x
				b = b + alpha *	G
			
			cost = ComputeCost(SynSample, a, b)
			cost_log.append(cost)
			
			if iteration > 1 :
				if abs(cost - prev_cost) < 10e-15 : 
					print ("Stop gradient descent at iteration", iteration)
					break
			prev_cost = cost
		print ("Total time", '%.5f' % (time.clock() - t0), "sec")
	elif mode == 2 :
		print ("Batch gradient descent")
		t0 = time.clock()
		for iteration in range(iteration_cnt) :
		#	print(iteration)
			GradientSum_a = 0.0
			GradientSum_b = 0.0
		
			for datanum in range(sampleNum):	
				#Compute gradient
				G = 0.0
				if SynSample[datanum].x < 10.0 and SynSample[datanum].x > -10.0 and SynSample[datanum].y < 10.0 and SynSample[datanum].y > -10.0 :
					G = (SynSample[datanum].y - (a * SynSample[datanum].x + b))
					GradientSum_a = GradientSum_a + G * SynSample[datanum].x
					GradientSum_b = GradientSum_b + G
		
			a = a + alpha * GradientSum_a
			b = b + alpha * GradientSum_b
			cost = ComputeCost(SynSample, a, b)
			cost_log.append(cost)
			# Stop gradient descent at some epsilon
		
			if iteration > 1 :
				if abs(cost - prev_cost) < 10e-15 : 
					print ("Stop gradient descent at iteration", iteration)
					break
			prev_cost = cost
		print ("Total time", '%.5f' % (time.clock() - t0), "sec")
	else :
		print ("Please input the correct GD mode")
		sys.exit(0)

	cost = ComputeCost(SynSample, a, b) 
	print("Converged cost", '%.5f' % cost)


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