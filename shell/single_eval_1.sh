export CUDA_VISIBLE_DEVICES=1
python ../eval.py rlbench.tasks=[reach_and_drag] \
    framework.logdir='/home/user/School/peract_l2r/logs/reach_and_drag' \
    framework.eval_from_eps_number=1000 

python ../eval.py rlbench.tasks=[stack_blocks_lv2] \
    framework.logdir='/home/user/School/peract_l2r/logs/stack_blocks_lv2' \
    framework.eval_from_eps_number=1000 

python ../eval.py rlbench.tasks=[put_money_in_safe] \
    framework.logdir='/home/user/School/peract_l2r/logs/put_money_in_safe' \
    framework.eval_from_eps_number=1000 

python ../eval.py rlbench.tasks=[place_wine_at_rack_location] \
    framework.logdir='/home/user/School/peract_l2r/logs/place_wine_at_rack' \
    framework.eval_from_eps_number=1000 

python ../eval.py rlbench.tasks=[put_groceries_in_cupboard] \
    framework.logdir='/home/user/School/peract_l2r/logs/put_groceries_in_cupboard' \
    framework.eval_from_eps_number=1000 

python ../eval.py rlbench.tasks=[push_buttons] \
    framework.logdir='/home/user/School/peract_l2r/logs/push_buttons' \
    framework.eval_from_eps_number=1000 

python ../eval.py rlbench.tasks=[stack_cups_lv2] \
    framework.logdir='/home/user/School/peract_l2r/logs/stack_cups_lv2' \
    framework.eval_from_eps_number=1000 

python ../eval.py rlbench.tasks=[open_drawer,put_item_in_drawer_lv2] \
    framework.logdir='/home/user/School/peract_l2r/logs/drawer_tasks' \
    framework.eval_from_eps_number=1000 