#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --time=00-01:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5g
#SBATCH --account=wellman98
​
echo "Job Id listed below:"
echo $SLURM_JOB_ID 
​
module load python cuda
module load clang 
module load gcc 

pip install cvxopt
pip install tensorflow
pip install matplotlib

cd ~/mfg_research/open_spiel/

g++ -o main open_spiel/tests/console_play_test.cc