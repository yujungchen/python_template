#!/usr/bin/python

import numpy as np
import random as rnd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
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


class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
		FancyArrowPatch.draw(self, renderer)


# Main function
if __name__ == '__main__':

	cost_log = [ ]
	cmdargs = str(sys.argv)

	# Number of samples
	sampleNum = int(sys.argv[1])
	iteration_cnt = int(sys.argv[2])

	data3D_x = [ ]
	data3D_y = [ ]
	data3D_z = [ ]
	SynSample3D = [ ]

	SampleArray = np.zeros((3, sampleNum))
	
	# Generate synthetic data
	for i in range(sampleNum):
		rx, ry, rz = Gaussian3DSampling(0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.8, 0.8, 0.8)
		data3D_x.append(rx)
		data3D_y.append(ry)
		data3D_z.append(rz)
		SampleArray[0, i] = rx
		SampleArray[1, i] = ry
		SampleArray[2, i] = rz
		SynSample3D.append(Sample3D(rx, ry, rz))


	# print(SampleArray)
	# Initialize parameters
	a = 0.0
	b = 0.0
	const = 0.0
	
	x = np.arange(-7, 7)
	y = np.arange(-7, 7)
	z = np.arange(-7, 7)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	for c, m in [('r', 'o')]:
		ax.scatter(data3D_x, data3D_y, data3D_z, c = c, marker = m)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')


	# Compute mean vector
	mean_vec = Sample3D(np.mean(data3D_x), np.mean(data3D_y), np.mean(data3D_z))
	#print("Mean", '%.5f' % mean_vec.x, '%.5f' % mean_vec.y, '%.5f' % mean_vec.z)
	mean_vector = np.array([[mean_vec.x], [mean_vec.y], [mean_vec.z]])

	# Compute Covariance Matrix
	Cov_Mat = np.zeros((3, 3))
	for datanum in range(sampleNum) :
		delta_x = data3D_x[datanum] - mean_vec.x
		delta_y = data3D_y[datanum] - mean_vec.y
		delta_z = data3D_z[datanum] - mean_vec.z
		delta = np.array([[delta_x], [delta_y], [delta_z]])
		Cov_Mat = Cov_Mat + delta.dot(delta.T)

	Cov_Mat = Cov_Mat / (sampleNum - 1)
	#print("Covariance Matrix", Cov_Mat)

	# Compute Eigenvector and Eigenvalue
	eig_val, eig_vec = np.linalg.eig(Cov_Mat.T)

	#for i in range(len(eig_val)) :
	#	eigvec_sc = eig_vec[:, i].reshape(1, 3).T
	#	print('Eigen Vector {} : \n{}'.format(i + 1, eigvec_sc))
	#	print('Eigen Value {} : {}'.format(i + 1, eig_val[i]))
	#	print(40 * '-')
		
	
	eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
	eig_pairs.sort()
	eig_pairs.reverse()

	#for i in eig_pairs:
	#	print(i[0], i[1])

	# Visualize Eigen Vector of Covaraince
	ax.plot([mean_vec.x], [mean_vec.y], [mean_vec.z], 'o', markersize = 10, color = 'blue', alpha = 0.5)
	for v in eig_vec.T :
		a = Arrow3D([mean_vec.x, v[0]], [mean_vec.y, v[1]], [mean_vec.z, v[2]], mutation_scale = 20, lw = 3, arrowstyle = "-|>", color = "g")
		ax.add_artist(a)

	#for ev in eig_vec:
	#	normp.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
	
	Proj_Mat = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
	# print(Proj_Mat)

	transformed = Proj_Mat.T.dot(SampleArray)

	plt.title('Sample Visualization')
	plt.grid(True)
	plt.axis([-7.0, 7.0, -7.0, 7.0])
	ax.set_zlim(-7, 7)

	plt.figure(2)
	plt.plot(transformed[0, :], transformed[1, :], 'o', markersize = 7, color = 'blue', alpha = 0.5, label = 'Sample')
	plt.axis([-10.0, 10.0, -10.0, 10.0])
	plt.xlabel('PC_1')	
	plt.ylabel('PC_2')
	plt.legend()
	plt.title('Transformed Samples using PCA')
	
	plt.grid(True)
	plt.show()