#import opensim as osim
#from osim.env import *
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import gym
import multiprocessing

from distributions import *

GAME = 'Pendulum-v0'
env = gym.make(GAME)
#env = RunEnv(visualize=False)

N_S = env.observation_space.shape[0] # 41 dim
N_A = env.action_space.shape[0]      # 18 dim
A_BOUND = [env.action_space.low, env.action_space.high] # 0 - 1

class PolicyEstimator():
	"""
	Policy Function approximator. Given a observation, returns probabilities
	over all possible actions.

	Args:
		num_outputs: Size of the action space.
		reuse: If true, an existing shared network will be re-used.
		trainable: If true we add train ops to the network.
			Actor threads that don't update their local models and don't need
			train ops would set this to false.
	"""
	def __init__(self, num_outputs, trainable=True):
		self.num_outputs = num_outputs
		self.use_state_dependent_std = True
		# Placeholders for our input
		# the env feedback observation
		self.states = tf.placeholder(shape=[None, N_S], dtype=tf.float32, name="states")
		# The TD target value
		self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name="advantage")
		# which action was selected
		self.actions = tf.placeholder(shape=[None, N_A], dtype=tf.float32, name="actions")

		batch_size = tf.shape(self.states)[0]

		# Policy net
		with tf.variable_scope("policy_net"):
			fc = self.build_policy_network(self.states)
			mu = tl.layers.DenseLayer(fc, n_units=self.num_outputs, act=tf.nn.tanh, name='mu')
			self.mu = mu.outputs
			self.sigma, dist_params= self._build_sigma(fc)

			self.dist = DiagNormal(dist_params)
			#normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

			with tf.name_scope('loss'):
				#log_prob = normal_dist.log_prob(self.actions)
				self.log_output_selected_action = self.dist.log_likelihood(self.actions)
				self.output_layer_entropy = self.dist.entropy()
				exp_v = self.log_output_selected_action * self.advantages
				#entropy = normal_dist.entropy()  # encourage exploration
				self.entropy = tf.reduce_sum(self.output_layer_entropy)
				self.exp_v = 0.01 * self.output_layer_entropy + exp_v
				self.loss = -tf.reduce_mean(self.exp_v)

			with tf.name_scope('choose_a'):  # use local params to choose action
				self.sample_action = self.dist.sample()
				#self.A = tf.clip_by_value(tf.squeeze(self.sample_action, axis=0), A_BOUND[0], A_BOUND[1])

			#self._build_gradient_ops(self.loss)

		if trainable:
			self.optimizer = tf.train.RMSPropOptimizer(0.0001)

	def choose_action(self, sess, s):  # run by a local
		s = s[np.newaxis, :]
		action, params = sess.run([self.sample_action, self.dist.params()], feed_dict={self.states: s})
		return action[0], params[0]

	def build_policy_network(self, X):
		"""
		Builds a network as described
		Args:
		X: Inputs
		add_summaries: If true, add layer summaries to Tensorboard.
		Returns:
		Final layer activations.
		"""

		# Three Fully connected layer
		X = tl.layers.InputLayer(X, name='p_input_layer')
		fc1 = tl.layers.DenseLayer(X, n_units=128,
								act=tf.nn.relu,
								name='p_fc1')
		fc2 = tl.layers.DenseLayer(fc1, n_units=64,
								act=tf.nn.relu,
								name='p_fc2')
		fc3 = tl.layers.DenseLayer(fc2, n_units=32,
								act=tf.nn.relu,
								name='p_fc3')

		return fc3

	def _build_sigma(self, fc):
		if self.use_state_dependent_std:
			self.sigma_hat = tl.layers.DenseLayer(
				fc, n_units=self.num_outputs, act=tf.identity, name='std2')
			self.sigma_hat = self.sigma_hat.outputs
			self.sigma2 = tf.log(1+tf.exp(self.sigma_hat))
			sigma = tf.sqrt(self.sigma2 + 1e-8)
			return sigma, tf.concat([self.mu, sigma], 1)
		else:
			self.log_sigma = tf.get_variable('log_sigma', self.mu.get_shape().as_list()[1],
				dtype=tf.float32, initializer=tf.random_uniform_initializer(0, 1))
			sigma = tf.expand_dims(tf.exp(self.log_sigma), 0)
			tiled_sigma = tf.tile(sigma, [tf.shape(self.mu)[0], 1])
			return sigma, tf.concat([self.mu, tiled_sigma], 1)

	def _clip_grads(self, grads):
		if self.clip_norm_type == 'ignore':
			return grads
		elif self.clip_norm_type == 'global':
			return tf.clip_by_global_norm(grads, self.clip_norm)[0]
		elif self.clip_norm_type == 'avg':
			return tf.clip_by_average_norm(grads, self.clip_norm)[0]
		elif self.clip_norm_type == 'local':
			return [tf.clip_by_norm(g, self.clip_norm)
					for g in grads]
		
	#def _build_gradient_ops(self, loss):
	#	self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
	#	self.flat_vars = utils.ops.flatten_vars(self.params)

	#	grads = tf.gradients(loss, self.params)
	#	self.get_gradients = self._clip_grads(grads)

		

class ValueEstimator():
	"""
	Value Function approximator. Returns a value estimator for a batch of observations.

	Args:
		reuse: If true, an existing shared network will be re-used.
		trainable: If true we add train ops to the network.
			Actor threads that don't update their local models and don't need
			train ops would set this to false.
	"""

	def __init__(self,trainable=True):
		# Placeholders for our input
		self.states = tf.placeholder(shape=[None, N_S], dtype=tf.float32, name="states")
		# The TD target value
		self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="targets")

		with tf.variable_scope("value_net"):
			fc = self.build_value_network(self.states)
			self.logits = tl.layers.DenseLayer(fc, n_units=1,
							act=tf.identity,
							name="logits")
			self.logits = self.logits.outputs

			self.losses = tf.squared_difference(self.logits, self.targets)
			self.loss = tf.reduce_sum(self.losses, name="loss")

			if trainable:
				self.optimizer = tf.train.RMSPropOptimizer(0.0001)

	def build_value_network(self, X):
		"""
		Builds a network as described
		Args:
		X: Inputs
		add_summaries: If true, add layer summaries to Tensorboard.
		Returns:
		Final layer activations.
		"""
		# Three Fully connected layer
		X = tl.layers.InputLayer(X, name='v_input_layer')
		fc1 = tl.layers.DenseLayer(X, n_units=128,
								act=tf.nn.relu,
								name='v_fc1')
		fc2 = tl.layers.DenseLayer(fc1, n_units=64,
								act=tf.nn.relu,
								name='v_fc2')
		fc3 = tl.layers.DenseLayer(fc2, n_units=32,
								act=tf.nn.relu,
								name='v_fc3')

		return fc3
