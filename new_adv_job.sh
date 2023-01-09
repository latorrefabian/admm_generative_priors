#!/bin/bash
python new_run_adversarial.py --seed 3 --cuda --n_iter 300  --n_images 2000 --n_points 500 --std 0.1
python new_run_adversarial.py --seed 3 --cuda --n_iter 300  --n_images 2000 --n_points 500 --std 0.2
python new_run_adversarial.py --seed 3 --cuda --n_iter 300  --n_images 2000 --n_points 500 --std 0.1 --restarts 2
python new_run_adversarial.py --seed 3 --cuda --n_iter 300  --n_images 2000 --n_points 500 --std 0.2 --restarts 2
