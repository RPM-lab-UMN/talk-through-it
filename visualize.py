# code to display a voxel grid from demo observation
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from helpers.utils import visualise_voxel
from voxel.voxel_grid import VoxelGrid
import numpy as np
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from agents import peract_bc
import os
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
from helpers.utils import create_obs_config

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.multiprocessing import set_start_method, get_start_method

try:
    if get_start_method() != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    print("Could not set start method to spawn")
    pass

from helpers.utils import stack_on_channel

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

def _preprocess_inputs(replay_sample, cams):
    obs, pcds = [], []
    for n in cams:
        rgb = stack_on_channel(replay_sample['%s_rgb' % n])
        pcd = stack_on_channel(replay_sample['%s_point_cloud' % n])
        
        rgb = _norm_rgb(rgb)

        obs.append([rgb, pcd]) # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd) # only pointcloud
    return obs, pcds

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # initialize voxelizer
    image_size = cfg.rlbench.camera_resolution
    device = 'cuda:0'
    vox_grid = VoxelGrid(
        coord_bounds=cfg.rlbench.scene_bounds,
        voxel_size=cfg.method.voxel_sizes[0],
        device=device,
        batch_size=1,
        feature_size=3,
        max_num_coords=np.prod(image_size) * len(cfg.rlbench.cameras)
    )
    os.environ['MASTER_ADDR'] = cfg.ddp.master_addr
    os.environ['MASTER_PORT'] = cfg.ddp.visualize_port
    cfg.rlbench.cameras = cfg.rlbench.cameras \
        if isinstance(cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    cams = cfg.rlbench.cameras
    obs_config = create_obs_config(cfg.rlbench.cameras,
                                   cfg.rlbench.camera_resolution)
    rank = 0
    world_size = 1
    dist.init_process_group("gloo",
                            rank=rank,
                            world_size=world_size)
    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks
    task_folder = task
    seed = 0
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % seed)
    cfg.replay.batch_size = 1 
    replay_buffer = peract_bc.launch_utils.create_replay(
        cfg.replay.batch_size, 
        cfg.replay.timesteps,
        replay_path if cfg.replay.use_disk else None,
        cfg.rlbench.cameras, 
        cfg.method.voxel_sizes,
        cfg.rlbench.camera_resolution)
    
    peract_bc.launch_utils.fill_multi_task_replay(
        cfg, obs_config, rank,
        replay_buffer, tasks, cfg.rlbench.demos,
        cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
        cfg.rlbench.cameras, cfg.rlbench.scene_bounds,
        cfg.method.voxel_sizes, cfg.method.bounds_offset,
        cfg.method.rotation_resolution, cfg.method.crop_augmentation,
        keypoint_method=cfg.method.keypoint_method)

    # initialize dataset
    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    dataset = wrapped_replay.dataset()
    train_data_iter = iter(dataset)

    # loop for number of demos
    for _ in range(cfg.rlbench.demos[0]):
        # sample from dataset
        batch = next(train_data_iter)
        lang_goal = batch['lang_goal'][0][0][0]
        batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}

        # preprocess observations
        obs, pcds = _preprocess_inputs(batch, cams)

        # flatten observations
        bs = obs[0][0].shape[0]
        pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcds], 1)

        image_features = [o[0] for o in obs]
        feat_size = image_features[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in image_features], 1)

        # tensorize scene bounds
        bounds = torch.tensor(cfg.rlbench.scene_bounds, device=device).unsqueeze(0)
        vox_grid.to(device)

        # voxelize!
        voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat, 
                                                            coord_features=flat_imag_features, 
                                                            coord_bounds=bounds)

        # swap to channels fist
        vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()

        # expert action voxel indicies
        vis_gt_coord = batch['trans_action_indicies'][:, -1, :3].int().detach().cpu().numpy()

        rotation_amount = 15
        show = True # shows trimesh scene
        # show = False # shows plt.imshow
        rendered_img = visualise_voxel(vis_voxel_grid[0],
                                    None,
                                    None,
                                    # vis_gt_coord[0],
                                    None,
                                    voxel_size=0.045,
                                    # voxel_size = 0.01,
                                    show=show,
                                    rotation_amount=np.deg2rad(rotation_amount),
                                    alpha=0.5)
        if not show:
            fig = plt.figure(figsize=(15, 15))
            plt.imshow(rendered_img)
            plt.axis('off')
            plt.show()

        print(f"Lang goal: {lang_goal}")

if __name__ == '__main__':
    main()