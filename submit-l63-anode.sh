#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=lorenz-anode-cpu-single-aug2
#SBATCH --account=flai
#SBATCH --output=lorenz-anode-cpu-single-aug2-%j-%N.out
#SBATCH --error=lorenz-anode-cpu-single-aug2-%j-%N.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --workdir=/p/tmp/maxgelbr/code/NeuralDELux.jl/
#SBATCH --mail-type=END
#SBATCH --mail-user=gelbrecht@pik-potsdam.de

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"


export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge
module load julia/1.9.1

julia /p/tmp/maxgelbr/code/NeuralDELux.jl/scripts/l63_augmented.jl $SLURM_JOB_NAME
