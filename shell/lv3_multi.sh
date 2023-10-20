# level 3 multi task experiment
export CUDA_VISIBLE_DEVICES=1

python ../fine_tune.py \
    rlbench.fine_tune_demos3=[5,5,5,5] \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_5_demos_2' \
    framework.start_weight=/home/user/School/peract_l2r/logs/lv1_to_lv2_10_demos/multi/PERACT_BC/seed2/weights/75000 \
    framework.start_seed=0

python ../fine_tune.py \
    rlbench.fine_tune_demos3=[5,5,5,5] \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_5_demos_2' \
    framework.start_weight=/home/user/School/peract_l2r/logs/lv1_to_lv2_10_demos/multi/PERACT_BC/seed2/weights/75000 \
    framework.start_seed=1

python ../fine_tune.py \
    rlbench.fine_tune_demos3=[5,5,5,5] \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_5_demos_2' \
    framework.start_weight=/home/user/School/peract_l2r/logs/lv1_to_lv2_10_demos/multi/PERACT_BC/seed2/weights/75000 \
    framework.start_seed=2


# evaluate each weight of each model
python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_5_demos_2' \
    framework.start_seed=0 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_5_demos_2' \
    framework.start_seed=1 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_5_demos_2' \
    framework.start_seed=2 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

# test each model
# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos_2' \
#     framework.start_seed=3 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[85000]

# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos_2' \
#     framework.start_seed=1 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[80000]

# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos_2' \
#     framework.start_seed=2 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[60000]