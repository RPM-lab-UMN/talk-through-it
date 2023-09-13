import os
import sys
# ARM_PATH = '/home/mshr/cliport_ndp/cliport_ndp'
# sys.path.append(ARM_PATH)

import numpy as np

import copy
import torch
from omegaconf import OmegaConf
from pyrep.objects import VisionSensor
from agents.peract_bc.launch_utils import create_agent
from helpers.clip.core.clip import tokenize
from helpers import utils
import logging



def dict_to_rlbench_observation(data, device):
	'''
	Args:
		data: dict of data from sensors. Contains:
			- cameras:
				- {cam}:
					- rgb: 			np.ndarray()
					- depth: 		np.ndarray()  # in meters
					- camera_intrinsics: 	np.ndarray()
					- camera_extrinsics: 	np.ndarray()
					TODO: figure out the array shapes. Is it [B, C, H, W]?
			- lang_goal: str
			- finger_positions: np.ndarray() [2,]
			- step: int  			# number of steps taken
			- episode_length: int  	# number of steps in the episode
	'''
	# cameras
	obs = {}
	for cam, d in data['cameras'].items():
		K = np.array(d['camera_intrinsics']).reshape(3,3)
		obs[f'{cam}_camera_intrinsics'] = torch.from_numpy(K).to(device)[None, None]

		E = np.array(d['camera_extrinsics'])
		obs[f'{cam}_camera_extrinsics'] = torch.from_numpy(E).to(device)[None, None]

		rgb = torch.from_numpy(d['rgb']).to(device)
		obs[f'{cam}_rgb'] = rgb.permute(2, 0, 1)[None, None]

		depth = torch.from_numpy(d['depth']).to(device)
		obs[f'{cam}_depth'] = depth[None, None]

		point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
			depth.clone().squeeze(0).cpu(), E, K)
		point_cloud = torch.from_numpy(point_cloud).to(device)
		obs[f'{cam}_point_cloud'] = point_cloud.permute(2, 0, 1)[None, None]

	# collision  TODO: remove from agent
	obs['ignore_collisions'] = torch.tensor([[[1.0]]], device=device)

	# language
	lang_goal_tokens = tokenize([data['lang_goal']])[0].numpy()
	lang_goal_tokens = torch.tensor([lang_goal_tokens], device=device).unsqueeze(0)
	obs['lang_goal_tokens'] = lang_goal_tokens

	# proprio TODO: update when we start doing manipulation.
	finger_positions = np.array(data['finger_positions'])
	gripper_open_amount = finger_positions[0] + finger_positions[1]
	gripper_open = (1.0 if (gripper_open_amount > 0.0385 + 0.0385) else 0.0)
	time = 1.0 - 2.0 * (data['step'] / float(data['episode_length'] - 1))
	low_dim_state = torch.tensor([[[gripper_open, 
									finger_positions[0],
									finger_positions[1],
									time]]])
	obs['low_dim_state'] = low_dim_state

	return obs


class PeractAgentInterface:

	def __init__(self, cfg):
		# TODO: pass only pointers to config here and load the config from logs.
		self.cfg = cfg
		assert 'seed_path' in self.cfg, 'Missing seed_path in cfg'
		if 'weight' not in self.cfg:
			self.cfg['weight'] = self._find_latest_trained_weights()

		# data
		if 'cameras_used' in cfg:
			self.cameras_used = cfg['cameras_used']
		else:
			self.cameras_used = ['front', 'left_shoulder', 'right_shoulder', 'overhead']
		self.camera_info = ['rgb', 'depth', 'camera_intrinsics', 'camera_extrinsics',]

		# agent
		if 'device' not in cfg:
			if torch.cuda.is_available(): cfg['device'] = 'cuda'
			else: cfg['device'] = 'cpu'
		self.device = torch.device(cfg['device'])
		self._load_agent()
		self.act_result = None


	def _check_data_structure(self, data):
		assert 'cameras' in data, f"Camera related data not provided"
		for cam in data['cameras']:
			assert cam in self.cameras_used, f"Camera {cam} not in expected cameras {self.cameras_used}"
			assert all([k in data['cameras'][cam] for k in self.camera_info]), f"Camera {cam} missing info {self.camera_info}"
		assert 'lang_goal' in data, f"Language goal not in data"
		
		# Currently we are not relying on joints states
		# assert 'joint_states' in data, f"Joint states not in data"
		if 'finger_positions' not in data:
			data['finger_positions'] = None

		assert 'step' in data, f"Data should indicate the number of steps taken"
		assert 'episode_length' in data, f"Data should indicate the expected episode length"


	def _load_agent(self):
		# load config
		cfg_path = os.path.join(self.cfg['seed_path'], 'config.yaml')
		cfg = OmegaConf.load(cfg_path)

		# load agent
		self.agent = create_agent(cfg)
		self.agent.build(training=False, device=self.device)

		# load pre-trained weights
		weights_path = os.path.join(self.cfg['seed_path'], 
			      				    'weights',
									str(self.cfg['weight']))
		logging.info("Loaded: " + weights_path)
		self.agent.load_weights(weights_path)


	def _find_latest_trained_weights(self):
		dir = os.path.join(self.cfg['seed_path'], 'weights')
		weights = [int(w) for w in os.listdir(dir)]
		weights.sort()
		return weights[-1]


	def check_and_mkdirs(self, dir_path):
		if not os.path.exists(dir_path):
			os.makedirs(dir_path, exist_ok=True)

	def _render_action_visual(self, result):  # result: ActResult (from yarr)
		voxel_grid 	= result.info['voxel_grid_depth0'].detach().cpu().numpy()[0]
		pred_q 		= result.info['q_depth0'].detach().cpu().numpy()[0]
		pred_t_idxs = result.info['voxel_idx_depth0'].detach().cpu().numpy()[0]
		voxel_render = utils.visualise_voxel(voxel_grid, 
											 pred_q, pred_t_idxs, None,
											 rotation_amount=np.deg2rad(0), 
											 voxel_size=0.045, alpha=0.5)
		return voxel_render

	def step(self, data):
		self._check_data_structure(data)
		observation = dict_to_rlbench_observation(data, self.device)
		result 		= self.agent.act(data['step'], observation,
											deterministic=True)
		action 		= result.action
		action_dict = {
			'position': action[0:3],
			'quaternion': {
				'x': action[3],
				'y': action[4],
				'z': action[5],
				'w': action[6],
			},
			'gripper_open': action[7],
			'ignore_collisions': action[8],	
		}

		render 		= self._render_action_visual(result)

		logging.info(f"Step: {data['step']} | Gripper Open: {action[7] > 0.99} | Ignore Collisions: {action[8] > 0.99}")

		return action_dict, render




if __name__ == '__main__':
	# Example setup:
	cfg = {
		'seed_path': 'path_to/seed2/',
		'weight': 39900,  # If unset, the latest weight will be used.
		'cameras_used': ['front', 'left_shoulder', 'right_shoulder', 'overhead'],
	}
	interface = PeractAgentInterface(cfg)
