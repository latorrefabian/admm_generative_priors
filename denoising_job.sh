#!/bin/bash
python run_denoising.py --seed 14 --restarts 2 --elu --cuda --dataset mnist --n_images 5 --norm -1 --n_iter 24000 --std 0.1 --alg gd adam linfeadmm
python run_denoising.py --seed 14 --restarts 2 --elu --cuda --dataset mnist --n_images 5 --norm 1 --n_iter 24000 --std 0.1 --alg gd adam l1eadmm
python run_denoising.py --seed 14 --restarts 2 --elu --cuda --dataset mnist --n_images 5 --norm -1 --n_iter 24000 --std 0.2 --alg gd adam linfeadmm
python run_denoising.py --seed 14 --restarts 2 --elu --cuda --dataset mnist --n_images 5 --norm 1 --n_iter 24000 --std 0.2 --alg gd adam l1eadmm
python run_denoising.py --seed 14 --restarts 2 --elu --cuda --dataset mnist --n_images 5 --norm -1 --n_iter 24000 --std 0.3 --alg gd adam linfeadmm
python run_denoising.py --seed 14 --restarts 2 --elu --cuda --dataset mnist --n_images 5 --norm 1 --n_iter 24000 --std 0.3 --alg gd adam l1eadmm

