#!/bin/bash
python run_finalexp_adversarial.py --n_iter 3000 --n_images 100 --seed 38 --std 0.1 --n_points 1500 --cuda
python run_finalexp_adversarial_robust.py --n_iter 3000 --n_images 100 --seed 38 --std 0.1 --n_points 1500 --cuda
