import os
import pickle
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from agents import peract_bc

def run_seed(rank,
             cfg: DictConfig,
             obs_config: ObservationConfig,
             cams,
             multi_task,
             seed,
             world_size,
             fine_tune = False) -> None:
    dist.init_process_group("gloo",
                            rank=rank,
                            world_size=world_size)

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks

    task_folder = task if not multi_task else 'multi'
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % seed)

    if fine_tune:
        batch_size = cfg.replay.batch_size - 1
    else:
        batch_size = cfg.replay.batch_size

    replay_buffer = peract_bc.launch_utils.create_replay(
        batch_size, 
        cfg.replay.timesteps,
        replay_path if cfg.replay.use_disk else None,
        cams, 
        cfg.method.voxel_sizes,
        cfg.rlbench.camera_resolution)

    peract_bc.launch_utils.fill_multi_task_replay(
        cfg, obs_config, rank,
        replay_buffer, tasks, cfg.rlbench.demos,
        cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
        cams, cfg.rlbench.scene_bounds,
        cfg.method.voxel_sizes, cfg.method.bounds_offset,
        cfg.method.rotation_resolution, cfg.method.crop_augmentation,
        keypoint_method=cfg.method.keypoint_method)
    
    if fine_tune:
        # create second replay buffer with demos for fine tuning
        fine_replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d_fine' % seed)
        fine_replay_buffer = peract_bc.launch_utils.create_replay(
            1, 
            cfg.replay.timesteps,
            fine_replay_path if cfg.replay.use_disk else None,
            cams, 
            cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)
        
        fine_tasks = cfg.rlbench.fine_tune_tasks
        fine_demos = cfg.rlbench.fine_tune_demos
        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            fine_replay_buffer, fine_tasks, fine_demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method='txt')

    agent = peract_bc.launch_utils.create_agent(cfg)

    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    fine_wrapped_replay = PyTorchReplayBuffer(fine_replay_buffer, num_workers=cfg.framework.num_workers) if fine_tune else None
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed)

    if cfg.framework.use_start_weight:
        start_weight = cfg.framework.start_weight
    else:
        start_weight = None

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        fine_wrapped_replay_buffer=fine_wrapped_replay,
        train_device=cfg.framework.gpu,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        start_weight=start_weight)

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()