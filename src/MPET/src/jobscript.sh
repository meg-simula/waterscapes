#!/bin/bash
# Job name:
#SBATCH --job-name=MPET
#
# Project:
#SBATCH --account=nn9279k
# Wall clock limit:
#SBATCH --time='1:00:00'
#
# Max memory usage per task:
#SBATCH --mem-per-cpu=4G
#
# Number of tasks (cores):
#SBATCH --nodes=1 --ntasks=8
#SBATCH --hint=compute_bound
#SBATCH --cpus-per-task=1

##SBATCH --partition=long
#SBATCH --output=MPET.out

## Set up job environment
source /cluster/bin/jobsetup

echo $SCRATCH

#source ~oyvinev/intro/hashstack/fenics-1.5.0.abel.gnu.conf
source ~oyvinev/fenics1.6/fenics1.6

# Expand pythonpath with locally installed packages
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python2.7/site-packages/

# Define what to do when job is finished (or crashes)
cleanup "mkdir -p /work/users/piersanti/MPET_output"
cleanup "cp -r $SCRATCH /work/users/piersanti/MPET_output"

# Copy necessary files to $SCRATCH
cp -r /usit/abel/u1/piersanti/MPET $SCRATCH

# Enter $SCRATCH and run job
cd $SCRATCH
cd MPET

mpirun --bind-to none python prova.py
