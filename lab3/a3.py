import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import logsoftmax
from utils import reduce_logsumexp
from collections import Counter

data = np.load('data2D.npy')

def getdist(X,Y):

	# distance of X^2 + Y^2 - 2XY	
	XX = tf.reshape(tf.reduce_sum(tf.multiply(X,X),1),[-1,1])
	YY = tf.reshape(tf.reduce_sum(tf.multiply(tf.transpose(Y),tf.transpose(Y)),0),[1,-1])
	XY = tf.scalar_mul(2.0,tf.matmul(X,tf.transpose(Y)))

	return XX + YY - XY 


def k_means(K,data,LEARNINGRATE,epochs,valid_start):

	train_data = data[:valid_start]
	valid_data = data[valid_start:]

	N = train_data.shape[0]
	D = train_data.shape[1]

	x = tf.placeholder("float32", [None,D])
	mu = tf.Variable(tf.random_normal([K,D],stddev=0.25))

	loss = tf.reduce_sum(tf.reduce_min(getdist(x,mu),1),0)

	adamop = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

	assign = tf.argmin(etdist(x,mu),1)

	init = tf.global_variables_initializer()

	train_loss = np.zeros(epochs)

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(epochs):
			L, _ = sess.run([loss, adamop], feed_dict={x:train_data})
			train_loss[epoch] = L 

		cluster_assign = sess.run(assign, feed_dict={x:train_data})

		valid_loss = sess.run(loss, feed_dict={x:valid_data})

	return train_loss, cluster_assign, valid_loss


# PART 1.1.2 ################################################################

# epochs = 600
# K = 3

# valid_loss1 = []
# valid_loss2 = []
# valid_loss3 = []

## Find best learing rate #

# for eta in [0.1, 0.01, 0.001]:

# 	validError, cluster_assign, _ = k_means(K, data, eta, epochs, len(data))
	
# 	if eta == 0.1:
# 		valid_loss1 = validError
# 	if eta == 0.01:
# 		valid_loss2 = validError
# 	if eta == 0.001:
# 		valid_loss3 = validError

# 	print eta

# plt.figure()
# plt.plot(range(epochs),valid_loss1[:],label="0.1",linewidth=0.75)
# plt.plot(range(epochs),valid_loss2[:],label="0.01",linewidth=0.75)
# plt.plot(range(epochs),valid_loss3[:],label="0.001",linewidth=0.75)
# plt.legend(loc='best')
# plt.title('Loss vs. Number of Epochs')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.show()

## Best Learning Rate = 0.01 #

# LEARNINGRATE = 0.01

# train_loss, cluster_assign, _ = k_means(K, data, LEARNINGRATE, epochs, len(data))
# print 'Minimum Training Loss: ', train_loss.min()
# print 'Minimum Validation Loss: ', valid_loss.min()

# plt.figure()
# plt.plot(range(epochs),train_loss)
# plt.title('Loss vs. Number of Epochs')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.show()

#############################################################################

# PART 1.1.3 ################################################################

# for k in range(1,6):
# 	print k 

# 	train_loss, cluster_assign, _ = k_means(k, data, LEARNINGRATE, epochs, len(data))
# 	print 'Minimum Training Loss: ', train_loss.min()

# 	samples = dict(Counter(cluster_assign))
# 	samples.update((x,y*100.0/data.shape[0]) for x,y in samples.items())
# 	print '% of points in each cluster: ', samples 

# 	# plot
# 	colors = ['c','r','g','m','y']
# 	for i in range(k):
# 		cluster_data = data[:len(cluster_assign)][cluster_assign==i].T
# 		if i == 0:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 1")
# 		if i == 1:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 2")
# 		if i == 2:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 3")
# 		if i == 3:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 4")
# 		if i == 4:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 5")
# 		plt.legend(loc='best')
# 	plt.show()

#############################################################################

# PART 1.1.4 ################################################################

# for k in range(1,6):
# 	print k

# 	train_loss, cluster_assign, valid_loss = k_means(k, data, LEARNINGRATE, epochs, 2*len(data)/3)
# 	print 'Minimum Training Loss: ', train_loss.min()
# 	print 'Minimum Validation Loss: ', valid_loss.min()


# MoG Stuff #################################################################

# Part 2.1.2 ################################################################

def log_pdf(X, mu, var):

	assert X.get_shape()[1] == mu.get_shape()[1]
	assert mu.get_shape()[0] == var.get_shape()[1]

	dist = getdist(X,mu)

	return -0.5*(tf.log(2*np.pi*var) + tf.multiply(dist, 1/var))

#############################################################################

# Part 2.1.3 ################################################################

def log_ZgivenX(X,mu,var,pi):
	log_pi_gauss = tf.log(pi) + log_pdf(X, mu, var)
	sum_log_pi_gauss = tf.reshape(reduce_logsumexp(log_pi_gauss,1),[-1,1])
	return log_pi_gauss - sum_log_pi_gauss

#############################################################################

# Part 2.2.2 ################################################################

def negLL(X, mu, var, pi_var):

	log_pi_gauss = tf.log(pi_var) + log_pdf(X, mu, var)
	sum_log_pi_gauss = tf.reshape(reduce_logsumexp(log_pi_gauss,1),[-1,1])

	tot_sum = tf.reduce_sum(sum_log_pi_gauss,0)

	return -tot_sum


def MoG(K, data, LEARNINGRATE, epochs, valid_start):

	train_data = data[:valid_start]
	valid_data = data[valid_start:]

	B = train_data.shape[0]
	D = train_data.shape[1]

	X = tf.placeholder("float32",shape=[None,D])
	mu = tf.Variable(tf.random_normal([K,D],stddev=0.25))
	var = tf.exp(tf.Variable(tf.random_normal([1,K],mean=0,stddev=0.25)))
	pi_var = tf.exp(logsoftmax(tf.Variable(tf.ones([1,K]))))

	L = negLL(X, mu, var, pi_var)

	adam_op = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)

	log_pi_gauss = tf.log(pi_var) + log_pdf(X, mu, var) 
	assign = tf.argmax(log_pi_gauss,1)

	init = tf.global_variables_initializer()

	loss_array = np.zeros(epochs)

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			loss, _ = sess.run([L,adam_op],feed_dict={X:train_data})
			loss_array[epoch] = loss

		clus_assign = sess.run(assign, feed_dict={X:train_data})
		mu_, var_, pi_var_ = sess.run([mu,var,pi_var])
		valid_loss = sess.run(L,feed_dict={X:valid_data})

	return loss_array, valid_loss, clus_assign, mu_, var_, pi_var_

def negFALL(X, mu, psi, W):
	Psi = tf.diag(psi)
	Psi_inv = tf.diag(1.0/psi)
	Sigma = Psi + tf.matmul(W, W, True, False)
	half_log_det = tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(Sigma))))

	diff = X-mu
	inv = tf.matrix_inverse(Sigma)
	inv_times_diff = tf.matmul(inv, diff, False, True)
	half_exponent = 0.5 * tf.reduce_sum(tf.matmul(diff, inv_times_diff))
	return half_log_det + half_exponent

def factorAnalysis (K, data, LEARNINGRATE, epochs):
	train_data = data['x']
	train_target = data['y']
	valid_data = data['x_valid']
	valid_target = data['y_valid']

	B = train_data.shape[0]
	D = train_data.shape[1]

	X = tf.placeholder("float32",shape=[None,D])
	mu = tf.Variable(tf.random_normal([D],stddev=0.25))
	psi = tf.exp(tf.Variable(tf.random_normal([D],mean=0,stddev=0.25)))
	W = tf.exp(tf.Variable(tf.random_normal([K, D],mean=0,stddev=0.25)))

	L = negFALL(X, mu, psi, W)

	adam_op = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)

	init = tf.global_variables_initializer()

	loss_array = np.zeros(epochs)

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			loss, _ = sess.run([L,adam_op],feed_dict={X:train_data})
			loss_array[epoch] = loss

		W_ = sess.run(W)
		valid_loss = sess.run(L,feed_dict={X:valid_data})
		print "Valid Loss: ", valid_loss
	
	return loss_array, valid_loss, W_


K = 3
epochs = 600
valid_start = len(data)

## Find the best learning rate #
#
# valid_loss1 = []
# valid_loss2 = []
# valid_loss3 = []
#
# for eta in [0.0075, 0.005, 0.0025]:
#
# 	train_loss, _, clus_assign, mu, var, pi_var, = MoG(K, data, eta, epochs, valid_start)
#	
# 	if eta == 0.0075:
# 		valid_loss1 = train_loss
# 	if eta == 0.005:
# 		valid_loss2 = train_loss
# 	if eta == 0.0025:
# 		valid_loss3 = train_loss
#
# 	print eta
#
# plt.figure()
# plt.plot(range(epochs),valid_loss1[:],label="0.0075",linewidth=0.75)
# plt.plot(range(epochs),valid_loss2[:],label="0.005",linewidth=0.75)
# plt.plot(range(epochs),valid_loss3[:],label="0.0025",linewidth=0.75)
# plt.legend(loc='best')
# plt.title('Loss vs. Number of Epochs')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.show()

## Best Learning Rate = 0.005

LEARNINGRATE = 0.005

# train_loss, _, clus_assign, mu, var, pi_var, = MoG(K, data, LEARNINGRATE, epochs, valid_start)
#
# plt.figure()
# plt.plot(range(epochs),train_loss[:],linewidth=0.75)
# plt.title('Loss vs. Number of Epochs')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.show()
#
# print 'Minimum Training Loss: ', train_loss.min()
# print 'Cluster means:\n', mu 
# print 'Cluster variances\n',var
# print 'Cluster pi values\n', pi_var

#############################################################################

# Part 2.2.3 ################################################################

# valid_start = 2*len(data)/3

# for k in range(1,6):
#
# 	train_loss, valid_loss, cluster_assign, mu, var, pi_var = MoG(k, data, LEARNINGRATE, epochs, valid_start)
#
# 	print k
#  	print 'Minimum Training Loss: ', train_loss[-1]
#  	print 'Minimum Validation Loss: ', valid_loss
#
#  	samples = dict(Counter(cluster_assign))
#  	#samples.update((x,y*100.0/data.shape[0]) for x,y in samples.items())
#
#  	print '% of points in each clusters: ', samples
#
# 	# plot
#
# 	colors = ['c','r','g','m','y']
#
# 	for i in range(k):
#
# 		print i 
#
# 		cluster_data = data[:len(cluster_assign)][cluster_assign==i].T
#
# 		if i == 0:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 1")
# 		if i == 1:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 2")
# 		if i == 2:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 3")
# 		if i == 3:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 4")
# 		if i == 4:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 5")
#
# 		plt.legend(loc='best')
#
# 	plt.show()

#############################################################################

# Part 2.2.4 ################################################################
# 
# data100 = np.load('data100D.npy')
# 
# k_eta = 0.01
# epochs = 600
# mog_eta = 0.005
# 
# valid_start = 2*len(data)/3
# 
# for k in range(1,11):
# 
# 	k_trainloss, _, k_validloss = k_means(k, data100, k_eta, epochs, valid_start)
#  	mog_trainloss, mog_validloss, _, _, _, _ = MoG(k, data100, mog_eta, epochs, valid_start)
# 
# 	print k
#  	
#  	print 'K_means Training Loss: ', k_trainloss[-1]
#  	print 'MoG Training Loss: ', mog_trainloss[-1]
# 
#  	print 'K_means Validation Loss: ', k_validloss
#  	print 'MoG Validation Loss: ', mog_validloss[0]
# 
#############################################################################

K = 4

tinymnist = np.load ("tinymnist.npz")
# 	trainData, trainTarget = data ["x"], data["y"]
#	validData, validTarget = data ["x_valid"], data ["y_valid"]
#	testData, testTarget = data ["x_test"], data ["y_test"]

# def factorAnalysis (K, data, LEARNINGRATE, epochs):
loss_array, valid_loss, W = factorAnalysis(K, tinymnist, LEARNINGRATE, 800)
# print W # W is 4 x 64
W = np.reshape(W, (K, 8, 8))
# print W

for i in range(K):
	plt.figure(i)
	plt.imshow(W[i])
	plt.title('Visualization of Row ' + str(i))
	plt.xlabel('')
	plt.ylabel('')
	plt.show()

