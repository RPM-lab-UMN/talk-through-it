# level 3 single task experiment
export CUDA_VISIBLE_DEVICES=1

# python ../fine_tune.py \
#     rlbench.fine_tune_tasks2=[open_drawer,put_item_in_drawer_lv2] \
#     rlbench.fine_tune_demos2=[10,10] \
#     rlbench.fine_tune_tasks3=[put_item_in_drawer_lv3] \
#     rlbench.fine_tune_demos3=[10] \
#     framework.logdir='/home/user/School/peract_l2r/logs/put_item_in_drawer_lv3_10_demos' \
#     framework.start_weight=/home/user/School/peract_l2r/logs/drawer_tasks/multi/PERACT_BC/seed0/weights/40000

# python ../fine_tune.py \
#     rlbench.fine_tune_tasks2=[stack_blocks_lv2] \
#     rlbench.fine_tune_demos2=[10] \
#     rlbench.fine_tune_tasks3=[stack_blocks_lv3] \
#     rlbench.fine_tune_demos3=[10] \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_blocks_lv3_10_demos' \
#     framework.start_weight=/home/user/School/peract_l2r/logs/stack_blocks_lv2/multi/PERACT_BC/seed0/weights/70000

# python ../fine_tune.py \
#     rlbench.fine_tune_tasks2=[push_buttons] \
#     rlbench.fine_tune_demos2=[10] \
#     rlbench.fine_tune_tasks3=[push_buttons_lv3] \
#     rlbench.fine_tune_demos3=[10] \
#     framework.logdir='/home/user/School/peract_l2r/logs/push_buttons_lv3_10_demos' \
#     framework.start_weight=/home/user/School/peract_l2r/logs/push_buttons/multi/PERACT_BC/seed0/weights/55000

# python ../fine_tune.py \
#     rlbench.fine_tune_tasks2=[stack_cups_lv2] \
#     rlbench.fine_tune_demos2=[10] \
#     rlbench.fine_tune_tasks3=[stack_cups_lv3] \
#     rlbench.fine_tune_demos3=[10] \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_cups_lv3_10_demos' \
#     framework.start_weight=/home/user/School/peract_l2r/logs/stack_cups_lv2/multi/PERACT_BC/seed0/weights/25000

# # evaluate each weight of each model
# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/put_item_in_drawer_lv3_10_demos' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=1000 \
#     framework.eval_type='missing'

# python ../eval.py \
#     rlbench.tasks=[stack_blocks] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_blocks_lv3_10_demos' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=1000 \
#     framework.eval_type='missing'

# python ../eval.py \
#     rlbench.tasks=[push_buttons_lv3] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/push_buttons_lv3_10_demos' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=1000 \
#     framework.eval_type='missing'

# python ../eval.py \
#     rlbench.tasks=[stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_cups_lv3_10_demos' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=1000 \
#     framework.eval_type='missing'

# test each model
# python ../eval.py \
#     rlbench.tasks=[put_item_in_drawer] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/put_item_in_drawer_lv3_10_demos' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[55000]

# python ../eval.py \
#     rlbench.tasks=[stack_blocks] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_blocks_lv3_10_demos' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[90000]

python ../eval.py \
    rlbench.tasks=[push_buttons_lv3] \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/push_buttons_lv3_10_demos' \
    framework.start_seed=0 \
    framework.eval_from_eps_number=2000 \
    framework.eval_type=[40000]

# python ../eval.py \
#     rlbench.tasks=[stack_cups] \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/stack_cups_lv3_10_demos' \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=2000 \
#     framework.eval_type=[25000]