import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from agents import peract_bc
import os
from helpers.utils import create_obs_config
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
import gc
import torch

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    logging.info('\n' + cfg_yaml)

    obs_config = create_obs_config(cfg.rlbench.cameras,
                                   cfg.rlbench.camera_resolution,
                                   cfg.method.name)

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks
    task_folder = task # if not multi_task else 'multi'
    seed = 0
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % seed)

    # create the replay buffer from RLBench demos
    replay_buffer = peract_bc.launch_utils.create_replay(
        cfg.replay.batch_size,
        cfg.replay.timesteps,
        replay_path if cfg.replay.use_disk else None,
        cfg.rlbench.cameras,
        cfg.method.voxel_sizes,
        cfg.rlbench.camera_resolution
    )
    rank = 0
    peract_bc.launch_utils.fill_multi_task_replay(
        cfg,
        obs_config,
        rank,
        replay_buffer,
        tasks,
        cfg.rlbench.demos,
        cfg.method.demo_augmentation,
        cfg.method.demo_augmentation_every_n,
        cfg.rlbench.cameras,
        cfg.rlbench.scene_bounds,
        cfg.method.voxel_sizes,
        cfg.method.bounds_offset,
        cfg.method.rotation_resolution,
        cfg.method.crop_augmentation,
        keypoint_method=cfg.method.keypoint_method
    )

    agent = peract_bc.launch_utils.create_agent(cfg)

    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed)

    world_size = cfg.ddp.num_devices
    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
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
        world_size=world_size)

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()