#!/bin/bash
#SBATCH --job-name=mfg_test
#SBATCH --time=00-01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5g
#SBATCH --account=wellman98
echo "Job Id listed below:"
echo $SLURM_JOB_ID 
​
module load python cuda
module load clang 
module load gcc

pip install clu
pip install cvxopt
pip install tensorflow
pip install matplotlib

cd ~/mfg_research/open_spiel/

python3 open_spiel/python/mfg/examples/mfg_fictitious_play.py --game_name='mfg_crowd_modelling_2d' --num_iterations=10 \
    --learning_rate=1e-2