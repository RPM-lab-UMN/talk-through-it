from calendar import c
import logging
from subprocess import REALTIME_PRIORITY_CLASS
import hydra
from omegaconf import DictConfig, OmegaConf
from agents import peract_bc
import os

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    logging.info('\n' + cfg_yaml)

    obs_config = create_obs_config(cfg.rlbench.cameras,
                                   cfg.rlbench.camera_resolution,
                                   cfg.method.name)

    task = cfg.rlbench.tasks[0]
    task_folder = task # if not multi_task else 'multi'
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % 0)

    # create the replay buffer from RLBench demos
    replay_buffer = peract_bc.launch_utils.create_replay(
        cfg.replay.batch_size,
        cfg.replay.timesteps,
        replay_path if cfg.replay.use_disk else None,
        cfg.rlbench.cameras,
        cfg.method.voxel_sizes,
        cfg.rlbench.camera_resolution
    )

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

if __name__ == "__main__":
    main()