export CUDA_VISIBLE_DEVICES=0
python eval.py rlbench.task_name='multi' \
    framework.logdir='/home/user/School/peract_l2r/logs/base_10_scripted'

python eval.py rlbench.task_name='multi' \
    framework.logdir='/home/user/School/peract_l2r/logs/base_10_scripted' \
    framework.start_seed=2 framework.eval_type=[60000]