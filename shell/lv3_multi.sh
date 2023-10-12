# level 3 multi task experiment
export CUDA_VISIBLE_DEVICES=0

python ../fine_tune.py \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
    framework.start_weight=/home/user/School/peract_l2r/logs/lv1_to_lv2_10_demos/multi/PERACT_BC/seed2/weights/75000 \
    framework.start_seed=0

python ../fine_tune.py \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
    framework.start_weight=/home/user/School/peract_l2r/logs/lv1_to_lv2_10_demos/multi/PERACT_BC/seed2/weights/75000 \
    framework.start_seed=1

python ../fine_tune.py \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
    framework.start_weight=/home/user/School/peract_l2r/logs/lv1_to_lv2_10_demos/multi/PERACT_BC/seed2/weights/75000 \
    framework.start_seed=2

# evaluate each weight of each model
python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
    framework.start_seed=0 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
    framework.start_seed=1 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
    framework.start_seed=2 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

# test each model
# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/put_item_in_drawer_lv3' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[40000]

# python ../eval.py \
#     rlbench.tasks=[stack_blocks] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_blocks_lv3' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[45000]

# python ../eval.py \
#     rlbench.tasks=[push_buttons_lv3] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/push_buttons_lv3' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[40000]

# python ../eval.py \
#     rlbench.tasks=[stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_cups_lv3' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[85000]