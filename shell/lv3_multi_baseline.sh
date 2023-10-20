# level 3 multi task experiment
export CUDA_VISIBLE_DEVICES=0

python ../train.py \
    ddp.master_port="'29503'" \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.demos=[10,10,10,10] \
    rlbench.demo_path=/home/user/School/peract_l2r/data_scripted \
    replay.path='/tmp/peract/replay2' \
    framework.logdir='/home/user/School/peract_l2r/logs/base_10_scripted_lv3' \
    framework.use_start_weight=False \
    framework.start_seed=1 \
    framework.load_existing_weights=True

python ../train.py \
    ddp.master_port="'29503'" \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.demos=[10,10,10,10] \
    replay.path='/tmp/peract/replay2' \
    rlbench.demo_path=/home/user/School/peract_l2r/data_scripted \
    framework.logdir='/home/user/School/peract_l2r/logs/base_10_scripted_lv3' \
    framework.use_start_weight=False \
    framework.start_seed=3

# evaluate each weight of each model
python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/base_10_scripted_lv3' \
    framework.start_seed=1 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

python ../eval.py \
    rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/base_10_scripted_lv3' \
    framework.start_seed=3 \
    framework.eval_from_eps_number=1000 \
    framework.eval_type='missing'

# test each model
# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos_2' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[60000]

# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
#     framework.start_seed=1 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[100000]

# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer,stack_blocks,push_buttons_lv3,stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv2_to_lv3_10_demos' \
#     framework.start_seed=2 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[95000]