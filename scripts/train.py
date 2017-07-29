#import opensim as osim
#from osim.env import *
import sys
import gym
import os
import numpy as np
import tensorflow as tf
import itertools
import cmd_options
import threading
import multiprocessing
from multiprocessing import Queue

from estimators import ValueEstimator, PolicyEstimator
from worker import Worker

GAME = 'Pendulum-v0'
env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]

def main(argv):
	args = cmd_options.get_arguments(argv)

	# Set the number of workers
	#NUM_WORKERS = 3
	NUM_WORKERS = multiprocessing.cpu_count()
	MODEL_DIR = args.model_dir
	args.task_queue = Queue()
	args.experience_queue = Queue()

	# Keeps track of the number of updates we've performed
	global_step = tf.Variable(0, name="global_step", trainable=False)

	# Global policy and value nets
	with tf.variable_scope("shared"):
		policy_net = PolicyEstimator(num_outputs=N_A)
		value_net = ValueEstimator()

	# Create worker graphs
	workers = []
	for worker_id in range(NUM_WORKERS):
		if worker_id == 0:
			visualize = True
		else:
			visualize = False

		worker = Worker(
			name="W_{}".format(worker_id),
			env=gym.make(GAME).unwrapped,
			policy_net=policy_net,
			value_net=value_net,
			discount_factor = 0.99,
			update_global_iter=args.update_global_iter,
			args=args)
		workers.append(worker)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()

		#print env.observation_space.shape[0], env.action_space.shape[0]

		# Load a previous checkpoint
		#saver.restore(sess, args.model_dir+'model.ckpt')

		# Start worker threads
		worker_threads = []

		for worker in workers:
			worker_fn = lambda: worker.run(sess, num_workers=NUM_WORKERS)
			t = threading.Thread(target=worker_fn)
			t.start()
			worker_threads.append(t)

		# Wait for all workers to finish
		coord.join(worker_threads)

if __name__ == '__main__':
	main(sys.argv[1:])
