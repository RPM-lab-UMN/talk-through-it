import gc
import logging
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from agents import peract_bc
from agents.command_classifier import CommandClassifier

from helpers import utils
from rlbench.backend import utils as rlbench_utils

from yarr.utils.rollout_generator import RolloutGenerator
from torch.multiprocessing import Process, Manager

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from clip import tokenize
from helpers.custom_rlbench_env import CustomRLBenchEnv
from yarr.agents.agent import Agent
import pickle
from rlbench.backend.const import *
from PIL import Image

class InteractiveEnv():
    def __init__(self, agent: Agent,
                 weightsdir: str = None,
                 classifier: CommandClassifier = None,
                 env_device: str = 'cuda:0'):
        self.agent = agent
        self.weightsdir = weightsdir
        self.classifier = classifier
        self.env_device = env_device

    def start(self, weight,
              env_config):

        eval_env = CustomRLBenchEnv(
            task_class=env_config[0],
            observation_config=env_config[1],
            action_mode=env_config[2],
            dataset_root=env_config[3],
            episode_length=env_config[4],
            headless=env_config[5],
            include_lang_goal_in_obs=env_config[6],
            time_in_state=env_config[7],
            record_every_n=env_config[8])

        self.eval_env = eval_env
        self.record_episodes(weight)

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def run_eval_interactive(self, weight):

        self.agent.build(training=False, device=self.env_device)

        logging.info('Launching env.')
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self.agent)

        env = self.eval_env
        env.eval = True
        env.launch()

        if not os.path.exists(self.weightsdir):
            raise Exception('No weights directory found.')

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self.weightsdir, str(weight))
            self.agent.load_weights(weight_path)

        # reset the task
        variation = 0
        eval_demo_seed = 1000 # TODO
        obs = env.reset_to_seed(variation, eval_demo_seed, interactive=True)
        prev_action = torch.zeros((1, 5)).to(self.env_device)
        prev_action[0, -1] = 1
        # replace the language goal with user input
        command = ''
        while command != 'quit':
            command = input("Enter a command: ")
            if command == 'reset':
                eval_demo_seed += 1
                obs = env.reset_to_seed(variation, eval_demo_seed, interactive=True)
                prev_action = torch.zeros((1, 5)).to(self.env_device)
                prev_action[0, -1] = 1
                continue
            # tokenize the command
            env._lang_goal = command
            tokens = tokenize([command]).numpy()
            # send the tokens to the classifier
            command_class = self.classifier.predict(tokens)
            # if command class is 1, use voxel transformer
            if command_class == 1:
                obs['lang_goal_tokens'] = tokens[0]
                self.agent.reset()
                timesteps = 1
                obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
                prepped_data = {k:torch.tensor([v], device=self.env_device) for k, v in obs_history.items()}

                act_result = self.agent.act(0, prepped_data,
                                        deterministic=eval)
                transition = env.step(act_result)
            else:
                # use l2a model
                text_embed = self.classifier.sentence_emb
                action, prev_action = self.classifier.l2a.get_action(prev_action, text_embed, obs)
                transition = env.step(action=action)
            env.env._scene.step()
            obs = dict(transition.observation)

    def check_and_make(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def save_demo(self, demo, example_path, obs_config, variation=0):

        # Save image data first, and then None the image data, and pickle
        left_shoulder_rgb_path = os.path.join(
            example_path, LEFT_SHOULDER_RGB_FOLDER)
        left_shoulder_depth_path = os.path.join(
            example_path, LEFT_SHOULDER_DEPTH_FOLDER)
        left_shoulder_mask_path = os.path.join(
            example_path, LEFT_SHOULDER_MASK_FOLDER)
        right_shoulder_rgb_path = os.path.join(
            example_path, RIGHT_SHOULDER_RGB_FOLDER)
        right_shoulder_depth_path = os.path.join(
            example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
        right_shoulder_mask_path = os.path.join(
            example_path, RIGHT_SHOULDER_MASK_FOLDER)
        overhead_rgb_path = os.path.join(
            example_path, OVERHEAD_RGB_FOLDER)
        overhead_depth_path = os.path.join(
            example_path, OVERHEAD_DEPTH_FOLDER)
        overhead_mask_path = os.path.join(
            example_path, OVERHEAD_MASK_FOLDER)
        wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
        wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
        wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
        front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
        front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
        front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

        self.check_and_make(left_shoulder_rgb_path)
        self.check_and_make(left_shoulder_depth_path)
        self.check_and_make(left_shoulder_mask_path)
        self.check_and_make(right_shoulder_rgb_path)
        self.check_and_make(right_shoulder_depth_path)
        self.check_and_make(right_shoulder_mask_path)
        self.check_and_make(overhead_rgb_path)
        self.check_and_make(overhead_depth_path)
        self.check_and_make(overhead_mask_path)
        self.check_and_make(wrist_rgb_path)
        self.check_and_make(wrist_depth_path)
        self.check_and_make(wrist_mask_path)
        self.check_and_make(front_rgb_path)
        self.check_and_make(front_depth_path)
        self.check_and_make(front_mask_path)

        for i, obs in enumerate(demo):
            # all rgbs
            left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
            right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
            overhead_rgb = Image.fromarray(obs.overhead_rgb)
            wrist_rgb = Image.fromarray(obs.wrist_rgb)
            # front_rgb = Image.fromarray(obs.front_rgb)
            left_shoulder_rgb.save(os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
            right_shoulder_rgb.save(os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
            overhead_rgb.save(os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
            wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
            # front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))

            # all depths
            if obs_config.left_shoulder_camera.depth:
                left_shoulder_depth = rlbench_utils.float_array_to_rgb_image(
                    obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
                left_shoulder_mask = Image.fromarray(
                    (obs.left_shoulder_mask * 255).astype(np.uint8))
                right_shoulder_depth = rlbench_utils.float_array_to_rgb_image(
                    obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
                right_shoulder_mask = Image.fromarray(
                    (obs.right_shoulder_mask * 255).astype(np.uint8))
                overhead_depth = rlbench_utils.float_array_to_rgb_image(
                    obs.overhead_depth, scale_factor=DEPTH_SCALE)
                overhead_mask = Image.fromarray(
                    (obs.overhead_mask * 255).astype(np.uint8))
                wrist_depth = rlbench_utils.float_array_to_rgb_image(
                    obs.wrist_depth, scale_factor=DEPTH_SCALE)
                wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
                front_depth = rlbench_utils.float_array_to_rgb_image(
                    obs.front_depth, scale_factor=DEPTH_SCALE)
                front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

                left_shoulder_depth.save(os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
                left_shoulder_mask.save(os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
                right_shoulder_depth.save(os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
                right_shoulder_mask.save(os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
                overhead_depth.save(os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
                overhead_mask.save(os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
                wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
                wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
                front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
                front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

            # We save the images separately, so set these to None for pickling.
            obs.left_shoulder_rgb = None
            obs.left_shoulder_depth = None
            obs.left_shoulder_point_cloud = None
            obs.left_shoulder_mask = None
            obs.right_shoulder_rgb = None
            obs.right_shoulder_depth = None
            obs.right_shoulder_point_cloud = None
            obs.right_shoulder_mask = None
            obs.overhead_rgb = None
            obs.overhead_depth = None
            obs.overhead_point_cloud = None
            obs.overhead_mask = None
            obs.wrist_rgb = None
            obs.wrist_depth = None
            obs.wrist_point_cloud = None
            obs.wrist_mask = None
            obs.front_rgb = None
            obs.front_depth = None
            obs.front_point_cloud = None
            obs.front_mask = None

        # Save the low-dimension data
        with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
            pickle.dump(demo, f)

        with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
            pickle.dump(variation, f)


    def record_episodes(self, weight):

        self.agent.build(training=False, device=self.env_device)

        logging.info('Launching env.')
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self.agent)

        env = self.eval_env
        env.eval = True
        env.launch()

        if not os.path.exists(self.weightsdir):
            raise Exception('No weights directory found.')

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self.weightsdir, str(weight))
            self.agent.load_weights(weight_path)

        # reset the task
        variation = 0
        eval_demo_seed = 1000 # TODO
        obs = env.reset_to_seed(variation, eval_demo_seed, interactive=True)
        prev_action = torch.zeros((1, 5)).to(self.env_device)
        prev_action[0, -1] = 1
        # create the episode folder
        episode_root = '/home/user/School/peract_l2r/data/pick_up/all_variations/episodes/'
        episode_idx = 0
        episode_dir = os.path.join(episode_root, 'episode' + str(episode_idx))
        # replace the language goal with user input
        command = ''
        demo = []
        while command != 'quit':
            command = input("Enter a command: ")
            if command == 'reset':
                # write the demo to file
                self.save_demo(demo, episode_dir, self.eval_env._observation_config)
                demo = []
                # update the episode directory
                episode_idx += 1
                episode_dir = os.path.join(episode_root, 'episode' + str(episode_idx))
                eval_demo_seed += 1
                obs = env.reset_to_seed(variation, eval_demo_seed, interactive=True)
                prev_action = torch.zeros((1, 5)).to(self.env_device)
                prev_action[0, -1] = 1
                continue
            # tokenize the command
            env._lang_goal = command
            tokens = tokenize([command]).numpy()
            # send the tokens to the classifier
            command_class = self.classifier.predict(tokens)
            # if command class is 1, use voxel transformer
            if command_class == 1:
                obs['lang_goal_tokens'] = tokens[0]
                self.agent.reset()
                timesteps = 1
                obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
                prepped_data = {k:torch.tensor([v], device=self.env_device) for k, v in obs_history.items()}

                act_result = self.agent.act(0, prepped_data,
                                        deterministic=eval)
                transition, demo_piece = env.record_step(act_result)
            else:
                # use l2a model
                text_embed = self.classifier.sentence_emb
                action, prev_action = self.classifier.l2a.get_action(prev_action, text_embed, obs)
                transition, demo_piece = env.record_step(action=action)
            env.env._scene.step()
            obs = dict(transition.observation)

            # extend the demo
            demo.extend(demo_piece)

def eval_seed(train_cfg,
              eval_cfg,
              logdir,
              cams,
              env_device,
              multi_task,
              seed,
              env_config) -> None:

    tasks = eval_cfg.rlbench.tasks
    agent = peract_bc.launch_utils.create_agent(train_cfg)
    weightsdir = os.path.join(logdir, 'weights')
    # get this file path
    cwd = os.path.dirname(os.path.realpath(__file__))
    print("cwd:", cwd)
    l2a_path = os.path.join(cwd, 'l2a.pt')
    classifier = CommandClassifier(input_size=1024, l2a_weights=l2a_path).to(env_device)
    # load classifier weights
    classifier_path = os.path.join(cwd, 'text_classifier.pt')
    classifier.load_state_dict(torch.load(classifier_path))

    # evaluate a specific checkpoint
    if type(eval_cfg.framework.interactive_weight) == int:
        weight_folders = [int(eval_cfg.framework.interactive_weight)]
        print("Weight:", weight_folders)
    else:
        raise Exception('Unknown eval type')
    
    num_weights_to_eval = np.arange(len(weight_folders))
    if len(num_weights_to_eval) == 0:
        logging.info("No weights to evaluate. Results are already available in eval_data.csv")
        sys.exit(0)

    interactive_env = InteractiveEnv(agent=agent, weightsdir=weightsdir, classifier=classifier, env_device=env_device)
    interactive_env.start(weight=weight_folders[0],
                          env_config=env_config)


@hydra.main(config_path='conf', config_name='eval')
def main(eval_cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(eval_cfg.framework.logdir,
                                eval_cfg.rlbench.task_name,
                                eval_cfg.method.name,
                                'seed%d' % start_seed)
    
    train_config_path = os.path.join(logdir, 'config.yaml')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            train_cfg = OmegaConf.load(f)
    else:
        raise Exception("Missing seed%d/config.yaml" % start_seed)
    
    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info('Using env device %s.' % str(env_device))

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.rlbench.cameras = eval_cfg.rlbench.cameras if isinstance(
        eval_cfg.rlbench.cameras, ListConfig) else [eval_cfg.rlbench.cameras]
    obs_config = utils.create_obs_config(eval_cfg.rlbench.cameras,
                                         eval_cfg.rlbench.camera_resolution)
    
    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    # single task or multi task
    if len(eval_cfg.rlbench.tasks) > 1:
        tasks = eval_cfg.rlbench.tasks
        multi_task = True
    
        task_classes = []
        for task in tasks:
            if task not in task_files:
                raise ValueError('Task %s not recognised!.' % task)
            task_classes.append(task_file_to_task_class(task))

        env_config = (task_classes,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      eval_cfg.framework.eval_episodes,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)
    else:
        task = eval_cfg.rlbench.tasks[0]
        multi_task = False

        if task not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = task_file_to_task_class(task)
        headless = False
        env_config = (task_class,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      headless,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)
        
    logging.info('Evaluating seed %d.' % start_seed)
    eval_seed(train_cfg,
              eval_cfg,
              logdir,
              eval_cfg.rlbench.cameras,
              env_device,
              multi_task, start_seed,
              env_config)

if __name__ == '__main__':
    main()