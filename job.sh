#!/bin/bash
#SBATCH -J rxnflow-sbdd-reward
#SBATCH -p a4000
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00

source /home/shwan/.bashrc

export WANDB_PROJECT='rxnflow_refactor_sbdd'
uv run ./script/train_pocket_conditional.py -o ./logs/sbdd-0204 --wandb
