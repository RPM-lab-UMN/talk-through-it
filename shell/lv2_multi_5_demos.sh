export CUDA_VISIBLE_DEVICES=1
# python ../fine_tune.py \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv1_to_lv2_5_demos' \
#     framework.start_seed=0

# python ../fine_tune.py \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv1_to_lv2_5_demos' \
#     framework.start_seed=1

# python ../fine_tune.py \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv1_to_lv2_5_demos' \
#     framework.start_seed=2

# evaluate to find best weights
# python ../eval.py \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv1_to_lv2_5_demos' \
#     framework.start_seed=1 \
#     framework.eval_from_eps_number=1000 \
#     framework.eval_type='missing'

# python ../eval.py \
#     rlbench.episode_length=25 \
#     framework.logdir='/home/user/School/peract_l2r/logs/lv1_to_lv2_5_demos' \
#     framework.start_seed=2 \
#     framework.eval_from_eps_number=1000 \
#     framework.eval_type='missing'

# test best weight for each
python ../eval.py \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv1_to_lv2_5_demos' \
    framework.start_seed=1 \
    framework.eval_from_eps_number=2000 \
    framework.eval_type=[95000]

python ../eval.py \
    rlbench.episode_length=25 \
    framework.logdir='/home/user/School/peract_l2r/logs/lv1_to_lv2_5_demos' \
    framework.start_seed=2 \
    framework.eval_from_eps_number=2000 \
    framework.eval_type=[85000]