import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.load("data2D.npy")

K = 3
D = data.shape[1]

print D

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

print mean, std

mu1 = tf.random_normal([K, 1], mean=mean[0], stddev=std[0])
mu2 = tf.random_normal([K, 1], mean=mean[1], stddev=std[1])

diff = tf.placeholder(tf.float32, [K, None, D])

mu = tf.concat(1, [mu1, mu2])
print mu
for i in range(K):
	diff = tf.concat(1, [diff, data - mu[i]])

x = tf.placeholder(tf.float32, [None, D])
x__ = x
 
loss = tf.reduce_sum(tf.reduce_min(tf.norm(x__ - mu, ord='euclidean', axis=1), axis=0))
# 
# print loss
