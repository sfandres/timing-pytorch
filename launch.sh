#!/bin/bash
## Shebang.


## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --ntasks=1                                  ## Number of tasks.
#SBATCH --ntasks-per-node=1                         ## Number of tasks to be invoked on each node.
#SBATCH --threads-per-core=1                        ## Restrict node selection to nodes with at least the specified number of threads per core.
#SBATCH --cpus-per-task=8                           ## Number of cpu-cores per task (>1 if multi-threaded tasks).
#SBATCH --gpus-per-node=1                           ## Number of GPUs on each node.
#SBATCH --mem=0                                     ## Real memory required per node (0: request all the memory on a node).
#SBATCH --exclusive                                 ## Job allocation can not share nodes with other running jobs.
#SBATCH --mail-type=ALL                             ## Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## User to receive the email notification.

## Catch Slurm environment variables.
job_id=${SLURM_JOB_ID}

## Create a string for email subject.
email_info="job_id=${job_id}"

## Send email when job begin (two options).
echo " " | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es

## Load virtual environment.
source /p/project/joaiml/hetgrad/anaconda3/etc/profile.d/conda.sh
conda activate lulc-conda

## Execute the Python script and pass the arguments.
echo "srun python3 timing_training_loop.py "$@""
srun python3 timing_training_loop.py "$@"

## Send email when job ends.
echo " " | /usr/bin/mail -s "Sbatch ${email_info} ended" sfandres@unex.es