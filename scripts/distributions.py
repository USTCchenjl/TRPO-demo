import tensorflow as tf
import numpy as np


class DiagNormal(object):
	'''
	Models Gaussian with Diagonal Covariance
	'''
	def __init__(self, params):
		self._params = params
		self.mu, self.sigma = tf.split(params, 2, 1)
		self.dim = tf.shape(self.mu)[1]

	def params(self):
		return self._params

	def sample(self):
		action = self.mu + self.sigma * tf.random_normal([self.dim])
		# action = tf.Print(action, [self.mu[0,0], self.sigma[0,0], action[0,0]], 'mu/sigma/action: ')
		return action

	def log_likelihood(self, x):
		return -tf.reduce_sum(
			0.5 * tf.square((x - self.mu) / (self.sigma + 1e-8))
			+ tf.log(self.sigma + 1e-8) + 0.5 * tf.log(2.0 * np.pi), axis=1)

	def entropy(self):
		return tf.reduce_sum(tf.log(self.sigma + 1e-8) + 0.5 * np.log(2 * np.pi * np.e), axis=1)

	def kl_divergence(self, params):
		mu_2, sigma_2 = tf.split(params, 2, 1)
		return tf.reduce_sum(
			(tf.square(sigma_2) + tf.square(mu_2 - self.mu)) / (2.0 * tf.square(self.sigma) + 1e-8) 
			+ tf.log(self.sigma/(sigma_2 + 1e-8) + 1e-8) - 0.5, axis=1)
