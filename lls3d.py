#!/usr/bin/python

import numpy as np
import random as rnd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import time

class Sample :
	def __init__(self, _x, _y) :
		self.x = _x
		self.y = _y

class Sample3D :
	def __init__(self, _x, _y, _z) :
		self.x = _x
		self.y = _y
		self.z = _z

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

# 3D gaussian random number
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


# Compute cost
def ComputeCost(Sample, a, b, const):
	cost = 0
	num = len(Sample)

	for idx in range(num):
		#print(idx, '%.3f' % x[idx], '%.3f' % y[idx])
		ErrorTerm = Sample[idx].z - a * Sample[idx].x - b * Sample[idx].y - const 
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
	mode = int(sys.argv[3])
	#epsilon = (1.0 / sampleNum)  / 100000.0
	#print ("Stopping criteria:", epsilon)
	prev_cost = 0

	data3D_x = [ ]
	data3D_y = [ ]
	data3D_z = [ ]
	SynSample3D = [ ]
	
	# Generate synthetic data
	for i in range(sampleNum):
		rx, ry, rz = Gaussian3DSampling(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.75, 0.75, 0.75)
		data3D_x.append(rx)
		data3D_y.append(ry)
		data3D_z.append(rz)
		SynSample3D.append(Sample3D(rx, ry, rz))


	# Initialize parameters
	a = 0.0
	b = 0.0
	const = 0.0
	alpha = 0.0001	#Learning rate
	x = np.arange(-7, 7)
	y = np.arange(-7, 7)
	z = np.arange(-7, 7)
	cost = ComputeCost(SynSample3D, a, b, const) 
	cost_log.append(cost)

	prev_cost = 0

	iteration = 0
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	for c, m in [('r', 'o')]:
		ax.scatter(data3D_x, data3D_y, data3D_z, c = c, marker = m)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	
	if mode == 1 :
		print ("Stochastic gradient descent")
		t0 = time.clock()
		for iteration in range(iteration_cnt) :
			for datanum in range(sampleNum):
				G = 0.0
				if SynSample3D[datanum].x < 10.0 and SynSample3D[datanum].x > -10.0 and SynSample3D[datanum].y < 10.0 and SynSample3D[datanum].y > -10.0 and SynSample3D[datanum].z < 10.0 and SynSample3D[datanum].z > -10.0:
					G = (SynSample3D[datanum].z - (a * SynSample3D[datanum].x + b * SynSample3D[datanum].y + const))

				a = a + alpha *	G * SynSample3D[datanum].x
				b = b + alpha *	G * SynSample3D[datanum].y
				const = const + alpha *	G
			
			cost = ComputeCost(SynSample3D, a, b, const)
			cost_log.append(cost)

			if iteration > 1 :
				if abs(cost - prev_cost) < 10e-9 : 
					print ("Stop gradient descent at iteration", iteration)
					break
			prev_cost = cost	
		print ("Total time", '%.5f' % (time.clock() - t0), "sec")
	elif mode == 2 :
		print ("Batch gradient descent")
		t0 = time.clock()
		for iteration in range(iteration_cnt) :
			GradientSum_a = 0.0
			GradientSum_b = 0.0
			GradientSum_c = 0.0
			
			for datanum in range(sampleNum) :
				G = 0.0
				if SynSample3D[datanum].x < 10.0 and SynSample3D[datanum].x > -10.0 and SynSample3D[datanum].y < 10.0 and SynSample3D[datanum].y > -10.0 and SynSample3D[datanum].z < 10.0 and SynSample3D[datanum].z > -10.0:
					G = SynSample3D[datanum].z - (a * SynSample3D[datanum].x + b * SynSample3D[datanum].y + const)
					GradientSum_a = GradientSum_a + G * SynSample3D[datanum].x
					GradientSum_b = GradientSum_b + G * SynSample3D[datanum].y
					GradientSum_c = GradientSum_c + G

			a = a + alpha * GradientSum_a
			b = b + alpha * GradientSum_b
			const = const + alpha * GradientSum_c
			cost = ComputeCost(SynSample3D, a, b, const)
			cost_log.append(cost)

			if iteration > 1 :
				if abs(cost - prev_cost) < 10e-9 : 
					print ("Stop gradient descent at iteration", iteration)
					break
			prev_cost = cost
		print ("Total time", '%.5f' % (time.clock() - t0), "sec")	
	else :
		print ("Please input the correct GD mode")
		sys.exit(0)



	z = a * x + b * y + const
	ax.plot(x, y, z)

	# Plot data and fitted line
	plt.title('Simple Linear Least Square')
	plt.grid(True)
	plt.axis([-7.0, 7.0, -7.0, 7.0])
	ax.set_zlim(-7, 7)
	#plt.axes().set_aspect('equal', 'datalim')

	plt.figure(2)
	plt.title('Cost')
	plt.plot(cost_log, '-')
	plt.grid(True)
	plt.show()