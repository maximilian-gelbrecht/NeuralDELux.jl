#!/bin/bash

#SBATCH --qos=gpushort
#SBATCH --job-name=l96-test
#SBATCH --account=flai
#SBATCH --output=l96-test-%j-%N.out
#SBATCH --error=l96-test-%j-%N.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --workdir=/p/tmp/maxgelbr/code/NeuralDELux.jl/
#SBATCH --mail-type=END
#SBATCH --mail-user=gelbrecht@pik-potsdam.de

echo "-----xw------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"


# Some initial setup
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge
module load julia/1.9.1

julia /p/tmp/maxgelbr/code/NeuralDELux.jl/scripts/l96.jl $SLURM_JOB_NAME
