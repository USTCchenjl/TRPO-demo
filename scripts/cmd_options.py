import argparse
import os.path

def get_arguments(argv):
	parser = argparse.ArgumentParser()

	# Basic parameters
	parser.add_argument('--max_global_steps', type=int, default=800)
	parser.add_argument('--model_dir', type=str, default='../weights/')
	parser.add_argument('--policy_rl', type=float, default=0.0001)
	parser.add_argument('--value_rl', type=float, default=0.001)
	parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor', dest='gamma')

	#clipping args
	parser.add_argument('--clip_loss', default=0.0, type=float, help='If bigger than 0.0, the loss will be clipped at +/-clip_loss', dest='clip_loss_delta')
	parser.add_argument('--clip_norm', default=40, type=float, help='If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm', dest='clip_norm')
	parser.add_argument('--clip_norm_type', default='global', help='Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)', dest='clip_norm_type')
	parser.add_argument('--rescale_rewards', action='store_true', help='If True, rewards will be rescaled (dividing by the max. possible reward) to be in the range [-1, 1]. If False, rewards will be clipped to be in the range [-REWARD_CLIP, REWARD_CLIP]', dest='rescale_rewards')
	parser.add_argument('--reward_clip_val', default=1.0, type=float, help='Clip rewards outside of [-REWARD_CLIP, REWARD_CLIP]', dest='reward_clip_val')

	#trpo args
	parser.add_argument('--num_epochs', default=1000, type=int, help='number of epochs for which to run TRPO', dest='num_epochs')
	parser.add_argument('--episodes_per_batch', default=20, type=int, help='number of episodes to batch for TRPO updates', dest='episodes_per_batch')
	parser.add_argument('--trpo_max_rollout', default=400, type=int, help='max rollout steps per trpo episode', dest='max_rollout')
	parser.add_argument('--cg_subsample', default=0.1, type=float, help='rate at which to subsample data for TRPO conjugate gradient iteration', dest='cg_subsample')
	parser.add_argument('--cg_damping', default=0.001, type=float, help='conjugate gradient damping weight', dest='cg_damping')   
	parser.add_argument('--max_kl', default=0.01, type=float, help='max kl divergence for TRPO updates', dest='max_kl')
	parser.add_argument('--td_lambda', default=0.97, type=float, help='lambda parameter for GAE', dest='td_lambda')

	args = parser.parse_args(argv)
	return args