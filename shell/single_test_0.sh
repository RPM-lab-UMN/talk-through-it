export CUDA_VISIBLE_DEVICES=0
python ../eval.py rlbench.tasks=[open_drawer] \
    framework.logdir='/home/user/School/peract_l2r/logs/open_drawer' \
    framework.eval_from_eps_number=2000 framework.eval_type=[60000]

python ../eval.py rlbench.tasks=[slide_block_to_color_target] \
    framework.logdir='/home/user/School/peract_l2r/logs/slide_block' \
    framework.eval_from_eps_number=2000 framework.eval_type=[65000]

python ../eval.py rlbench.tasks=[sweep_to_dustpan_of_size] \
    framework.logdir='/home/user/School/peract_l2r/logs/sweep_to_dustpan' \
    framework.eval_from_eps_number=2000 framework.eval_type=[100000]

python ../eval.py rlbench.tasks=[turn_tap] \
    framework.logdir='/home/user/School/peract_l2r/logs/turn_tap' \
    framework.eval_from_eps_number=2000 framework.eval_type=[90000]

python ../eval.py rlbench.tasks=[put_item_in_drawer_lv2] \
    framework.logdir='/home/user/School/peract_l2r/logs/put_item_in_drawer_lv2' \
    framework.eval_from_eps_number=2000 framework.eval_type=[25000]

python ../eval.py rlbench.tasks=[close_jar] \
    framework.logdir='/home/user/School/peract_l2r/logs/close_jar' \
    framework.eval_from_eps_number=2000 framework.eval_type=[45000]