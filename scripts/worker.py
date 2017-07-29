import sys
import gym
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
from multiprocessing import Process

from estimators import *

MAX_GLOBAL_EP = 100000
GLOBAL_EP = 0
GLOBAL_RUNNING_R = []

def make_copy_params_op(v1_list, v2_list):
	"""
	Creates an operation that copies parameters from variable in v1_list to variables in v2_list. dest_net=v1, v1=v2
	"""
	update_ops = [l_p.assign(g_p) for l_p, g_p in zip(v1_list, v2_list)]

	return update_ops

class Worker(object):
	"""
	An TRPO worker thread. Runs episodes locally and updates global shared value and policy nets.

	Args:
		name: A unique name for this worker
		env: The environment used by this worker
		policy_net: Instance of the globally shared policy net
		value_net: Instance of the globally shared value net
		discount_factor: Reward discount factor
		max_global_steps: If set, stop coordinator when global_counter > max_global_steps
	"""
	def __init__(self, name, env, policy_net, value_net, discount_factor, update_global_iter, args):
		self.name = name
		self.discount_factor = discount_factor
		#self.max_global_steps = max_global_steps
		self.update_global_iter = update_global_iter
		self.global_step = tf.train.get_global_step()
		self.shared_policy_net = policy_net
		self.shared_value_net = value_net
		self.env = env

		#TRPO parameters
		self.gamma = args.gamma
		self.td_lambda = args.td_lambda
		self.batch_size = 64
		self.max_cg_iters = 10
		self.num_epochs = args.num_epochs
		self.cg_damping = args.cg_damping
		self.cg_subsample = args.cg_subsample
		self.max_kl = args.max_kl
		self.max_rollout = args.max_rollout
		self.episodes_per_batch = args.episodes_per_batch
		self.experience_queue = args.experience_queue
		self.task_queue = args.task_queue

		shared_policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared/policy_net')
		shared_value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared/value_net')

		# Create local policy/value nets that are not updated asynchronously
		with tf.variable_scope(name):
			self.policy_net = PolicyEstimator(policy_net.num_outputs)
			self.value_net = ValueEstimator()
			self.local_policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/policy_net')
			self.local_value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/value_net')
			#self.p_grads = tf.gradients(self.policy_net.loss, self.local_policy_params)
			self.v_grads = tf.gradients(self.value_net.loss, self.local_value_params)

			self.policy_flat_vars = self.flatten_vars(self.local_policy_params)
		# Op to copy params from global policy/valuenets
		self.restore_v_params = [make_copy_params_op(self.local_value_params, shared_value_params)]
		self.restore_p_params = [make_copy_params_op(self.local_policy_params, shared_policy_params)]
		self.restore_params_op = [make_copy_params_op(self.local_policy_params, shared_policy_params), 
								make_copy_params_op(self.local_value_params, shared_value_params) ] #restore: shared to local

		self.copy_params_op = [make_copy_params_op(shared_policy_params, self.local_policy_params), 
								make_copy_params_op(shared_value_params, self.local_value_params) ] #copy/save: local to shared
		self.copy_p_params = [make_copy_params_op(shared_policy_params, self.local_policy_params)]
		self.copy_v_params = [make_copy_params_op(shared_value_params, self.local_value_params) ]

		self._setup_shared_memory_ops()
		self._build_ops()
		
		self.vnet_train_op = self.value_net.optimizer.apply_gradients(zip(self.v_grads, self.local_value_params), global_step=self.global_step)

		self.state = None

		if self.name == 'W_0':
			self.saver = tf.train.Saver()

	def _setup_shared_memory_ops(self):
		# Placeholders for shared memory vars
		self.params_ph = []
		for p in self.local_policy_params:
			self.params_ph.append(tf.placeholder(tf.float32, 
				shape=p.get_shape(), 
				name="shared_memory_for_{}".format(
					(p.name.split("/", 1)[1]).replace(":", "_"))))
			
		# Ops to sync net with shared memory vars
		self.sync_with_shared_memory = []
		for i in xrange(len(self.local_policy_params)):
			self.sync_with_shared_memory.append(
				self.local_policy_params[i].assign(self.params_ph[i]))

	def _build_ops(self):
		self.dist_params = self.policy_net.dist.params()
		num_params = self.dist_params.get_shape().as_list()[1]
		self.old_params = tf.placeholder(tf.float32, shape=[None, num_params], name='old_params')

		selected_prob = tf.exp(self.policy_net.log_output_selected_action)
		old_dist = self.policy_net.dist.__class__(self.old_params)
		old_selected_prob = tf.exp(old_dist.log_likelihood(self.policy_net.actions))

		self.theta = self.policy_flat_vars
		self.policy_loss = -tf.reduce_mean(tf.multiply(
			self.policy_net.advantages,
			selected_prob / old_selected_prob
		))
		self.pg = self.flatten_vars(
			tf.gradients(self.policy_loss, self.local_policy_params))

		self.kl = tf.reduce_mean(self.policy_net.dist.kl_divergence(self.old_params))
		self.kl_firstfixed = tf.reduce_mean(self.policy_net.dist.kl_divergence(
			tf.stop_gradient(self.dist_params)))

		kl_grads = tf.gradients(self.kl_firstfixed, self.local_policy_params)
		flat_kl_grads = self.flatten_vars(kl_grads)

		self.pg_placeholder = tf.placeholder(tf.float32, shape=self.pg.get_shape().as_list(), name='pg_placeholder')
		self.fullstep, self.neggdotstepdir = self._conjugate_gradient_ops(
			-self.pg_placeholder, flat_kl_grads, max_iterations=self.max_cg_iters)


	def _conjugate_gradient_ops(self, pg_grads, kl_grads, max_iterations=10, residual_tol=1e-10):
		'''
		Construct conjugate gradient descent algorithm inside computation graph for improved efficiency
		'''
		i0 = tf.constant(0, dtype=tf.int32)
		loop_condition = lambda i, r, p, x, rdotr: tf.logical_and(
			tf.greater(rdotr, residual_tol), tf.less(i, max_iterations))


		def body(i, r, p, x, rdotr):
			fvp = self.flatten_vars(tf.gradients(
				tf.reduce_sum(tf.stop_gradient(p)*kl_grads),
				self.local_policy_params))

			z = fvp + self.cg_damping * p

			alpha = rdotr / (tf.reduce_sum(p*z) + 1e-8)
			x += alpha * p
			r -= alpha * z

			new_rdotr = tf.reduce_sum(r*r)
			beta = new_rdotr / (rdotr + 1e-8)
			p = r + beta * p

			new_rdotr = tf.Print(new_rdotr, [i, new_rdotr], 'Iteration / Residual: ')

			return i+1, r, p, x, new_rdotr

		_, r, p, stepdir, rdotr = tf.while_loop(
			loop_condition,
			body,
			loop_vars=[i0,
					   pg_grads,
					   pg_grads,
					   tf.zeros_like(pg_grads),
					   tf.reduce_sum(pg_grads*pg_grads)])

		fvp = self.flatten_vars(tf.gradients(
			tf.reduce_sum(tf.stop_gradient(stepdir)*kl_grads),
			self.local_policy_params))

		shs = 0.5 * tf.reduce_sum(stepdir*fvp)
		lm = tf.sqrt((shs + 1e-8) / self.max_kl)
		fullstep = stepdir / lm
		neggdotstepdir = tf.reduce_sum(pg_grads*stepdir) / lm

		return fullstep, neggdotstepdir

	def flatten_vars(self, var_list):
		return tf.concat([
			tf.reshape(v, [np.prod(v.get_shape().as_list())])
			for v in var_list
			], 0)

	def _value_net_predict(self, state, sess):
		feed_dict = { self.value_net.states: state }
		preds = sess.run(self.value_net.logits, feed_dict)
		return preds[:, 0]

	def _compute_gae(self, rewards, values, next_val):
		values = values + [next_val]
		size = len(rewards)
		adv_batch = list()
		td_i = 0.0

		for i in reversed(xrange(size)):
			td_i = rewards[i] + self.gamma*values[i+1] - values[i] + self.td_lambda*self.gamma*td_i 
			adv_batch.append(td_i)

		adv_batch.reverse()
		return adv_batch

	def assign_policy_vars(self, sess, params):
		feed_dict = {}
		offset = 0

		for i, var in enumerate(self.local_policy_params):
			shape = var.get_shape().as_list()
			size = np.prod(shape)
			if type(params) == list:
				feed_dict[self.params_ph[i]] = params[i]
			else:
				feed_dict[self.params_ph[i]] = \
					params[offset:offset+size].reshape(shape)
			offset += size

		sess.run(self.sync_with_shared_memory, feed_dict=feed_dict)

	def fit_valuenet(self, data, sess, mix_old=.9):
		data_size = len(data['state'])
		proc_state = data['state']
		print 'diffs', (data['mc_return'] - data['values']).mean()
		target = (1-mix_old)*data['mc_return'] + mix_old*data['values']
		#grads = [np.zeros(g.get_shape().as_list(), dtype=np.float32) for g in self.v_grads]

		#permute data in minibatches so we don't introduce bias
		perm = np.random.permutation(data_size)
		for start in range(0, data_size, self.batch_size):
			end = start + np.minimum(self.batch_size, data_size-start)
			batch_idx = perm[start:end]
			feed_dict={
				self.value_net.states: proc_state[batch_idx],
				self.value_net.targets: target[batch_idx]
			}
			#output_i = sess.run(self.v_grads, feed_dict=feed_dict)
			
			#for i, g in enumerate(output_i):
			#	grads[i] += g * (end-start)/float(data_size)

			sess.run(self.vnet_train_op, feed_dict=feed_dict) #train_shared
			sess.run(self.copy_v_params)

	def run_minibatches(self, sess, data, *ops):
		outputs = [np.zeros(op.get_shape().as_list(), dtype=np.float32) for op in ops]

		data_size = len(data['state'])
		for start in range(0, data_size, self.batch_size):
			end = start + np.minimum(self.batch_size, data_size-start)
			feed_dict={
				self.policy_net.states:           data['state'][start:end],
				self.policy_net.actions: 		  data['action'][start:end],
				self.policy_net.advantages:       data['reward'][start:end],
				self.old_params:                  data['pi'][start:end]
			}
			for i, output_i in enumerate(sess.run(ops, feed_dict=feed_dict)):
				outputs[i] += output_i * (end-start)/float(data_size)

		return outputs

	def linesearch(self, sess, data, x, fullstep, expected_improve_rate):
		accept_ratio = .1
		backtrack_ratio = .7
		max_backtracks = 15
	
		fval = self.run_minibatches(sess, data, self.policy_loss)

		for (_n_backtracks, stepfrac) in enumerate(backtrack_ratio**np.arange(max_backtracks)):
			xnew = x + stepfrac * fullstep
			self.assign_policy_vars(sess, xnew)
			newfval, kl = self.run_minibatches(sess, data, self.policy_loss, self.kl)

			improvement = fval - newfval

			expected_improve = expected_improve_rate * stepfrac
			ratio = improvement / expected_improve
			# if ratio > accept_ratio and improvement > 0:
			if kl < self.max_kl and improvement > 0:
				print 'Update'
				return xnew
		print 'No update'
		return x

	def update_grads(self, data, sess):
		#we need to compute the policy gradient in minibatches to avoid GPU OOM errors on Atari
		print 'fitting valuenet...'
		self.fit_valuenet(data, sess)

		normalized_advantage = (data['advantage'] - data['advantage'].mean())/(data['advantage'].std() + 1e-8)
		data['reward'] = normalized_advantage

		print 'running policy gradient...'
		pg = self.run_minibatches(sess, data, self.pg)[0]

		data_size = len(data['state'])
		subsample = np.random.choice(data_size, int(data_size*self.cg_subsample), replace=False)
		feed_dict={
			self.policy_net.states:           data['state'][subsample],
			self.policy_net.actions: 		  data['action'][subsample],
			self.policy_net.advantages:       data['reward'][subsample],
			self.old_params:                  data['pi'][subsample],
			self.pg_placeholder:              pg
		}

		print 'running conjugate gradient descent...'
		theta_prev, fullstep, neggdotstepdir = sess.run(
			[self.theta, self.fullstep, self.neggdotstepdir], feed_dict=feed_dict)

		print 'running linesearch...'
		new_theta = self.linesearch(sess, data, theta_prev, fullstep, neggdotstepdir)
		self.assign_policy_vars(sess, new_theta)

		return sess.run(self.kl, feed_dict)

	def _run_master(self, sess):
		for epoch in range(self.num_epochs):
			sess.run(self.restore_params_op)
			data = {
				'state':     list(),
				'pi':        list(),
				'action':    list(),
				'reward':    list(),
				'advantage': list(),
				'mc_return': list(),
				'values':    list(),
			}
			#launch worker tasks
			for i in xrange(self.episodes_per_batch):
				self.task_queue.put(i)

			#collect worker experience
			episode_rewards = list()
			for _ in xrange(self.episodes_per_batch):
				print 'experience_get'
				worker_data, reward = self.experience_queue.get()
				episode_rewards.append(reward)

				values = self._value_net_predict(worker_data['state'], sess)
				advantages = self._compute_gae(worker_data['reward'], values.tolist(), 0)
				# advantages = worker_data['mc_return'] - values
				worker_data['values'] = values
				worker_data['advantage'] = advantages
				for key, value in worker_data.items():
					data[key].extend(value)

			kl = self.update_grads({
				k: np.array(v) for k, v in data.items()}, sess)
			sess.run(self.copy_params_op)

			mean_episode_reward = np.array(episode_rewards).mean()
			print 'Mean Reward: ', mean_episode_reward
			#logger.info('Epoch {} / Mean KL Divergence {} / Mean Reward {} / Experience Time {:.2f}s / Training Time {:.2f}s'.format(
			#	epoch+1, kl, mean_episode_reward, t1-t0, t2-t1))
	def _run_worker(self, sess):
		while True:
			print 'task_get'
			signal = self.task_queue.get()
			if signal == 'EXIT':
				print 'EXIT'
				break

			sess.run(self.restore_params_op)
			self.state = self.env.reset()

			data = {
				'state':     list(),
				'pi':        list(),
				'action':    list(),
				'reward':    list(),
			}
			episode_over = False
			accumulated_rewards = list()
			while not episode_over and len(accumulated_rewards) < self.max_rollout:
				if self.name == 'W_1':
					self.env.render()
				action, pi = self.policy_net.choose_action(sess, self.state)
				next_state, reward, episode_over, info = self.env.step(action)
				reward /= 10
				accumulated_rewards.append(reward)

				data['state'].append(self.state)
				data['pi'].append(pi)
				data['action'].append(action)
				data['reward'].append(reward)

				self.state = next_state

			mc_returns = list()
			running_total = 0.0
			for r in reversed(accumulated_rewards):
				running_total = r + self.gamma*running_total
				mc_returns.insert(0, running_total)

			data['mc_return'] = mc_returns
			episode_reward = sum(accumulated_rewards)
			print('T{} / Episode Reward {}'.format(
				self.name, episode_reward))

			self.experience_queue.put((data, episode_reward))

	def run(self, sess, num_workers):
		try:
			if self.name == 'W_0':
				self._run_master(sess)
				for _ in xrange(num_workers):
					self.task_queue.put('EXIT')
			else:
				self._run_worker(sess)
		except:
			os._exit(0)
