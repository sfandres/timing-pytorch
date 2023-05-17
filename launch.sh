#!/bin/bash
## Shebang.


## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --ntasks=1                                  ## Number of tasks.
#SBATCH --ntasks-per-node=1                         ## Number of tasks to be invoked on each node.
#SBATCH --cpus-per-task=8                           ## Number of cpu-cores per task (>1 if multi-threaded tasks).
#SBATCH --gpus-per-node=1                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --mail-type=ALL                             ## (not working) Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## (not working) User to receive the email notification.

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
/usr/bin/mail -a timing_${job_id}.out -s "Sbatch ${email_info} ended" sfandres@unex.es