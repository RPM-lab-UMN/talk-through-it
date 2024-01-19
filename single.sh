export CUDA_VISIBLE_DEVICES=1
python fine_tune.py rlbench.fine_tune_tasks=[open_drawer] \
    framework.logdir='/home/user/School/peract_l2r/logs/open_drawer' \
    framework.gpu=0
