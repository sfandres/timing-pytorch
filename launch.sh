#!/bin/bash
## Shebang.


## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --ntasks=1                                  ## Number of tasks.
#SBATCH --ntasks-per-node=1                         ## Number of tasks to be invoked on each node.
#SBATCH --cpus-per-task=1                           ## Number of cpu-cores per task (>1 if multi-threaded tasks).
#SBATCH --time=00:10:00                             ## Job duration.
##SBATCH --cpus-per-gpu=1                           ## Number of cpu-cores per task (>1 if multi-threaded tasks).
##SBATCH --threads-per-core=1                       ## Restrict node selection to nodes with at least the specified number of threads per core.
#SBATCH --gpus-per-node=1                           ## Min. number of GPUs on each node.
#SBATCH --mem=4G                                    ## Real memory required per node (0: request all the memory on a node).
#SBATCH --job-name=timing_%j                        ## Name of the job.
#SBATCH --output=timing_%j.out                      ## Output file.
#SBATCH --exclusive                                 ## Job allocation can not share nodes with other running jobs.
#SBATCH --mail-type=ALL                             ## Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## User to receive the email notification.


## Catch Slurm environment variables.
job_id=$SLURM_JOB_ID

## Create a string for email subject.
email_info="job_id=$job_id"

## Send email when job begin (two options).
## echo " " | /usr/bin/mail -s "Sbatch $email_info began" sfandres@unex.es

## Load virtual environment.
source /p/project/joaiml/hetgrad/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-timing-conda

## Juelich configuration.
## export CUDA_VISIBLE_DEVICES=0
## export OMP_NUM_THREADS=1
## if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
##     export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
## fi
## echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
## echo "OMP_NUM_THREADS:      $OMP_NUM_THREADS"

## Execute the Python script and pass the arguments.
## command="kernprof -l timing_training_loop.py "$@""  ## include @profile in code.
command="python3 timing_training_loop.py "$@""
echo "Executed command: $command"
echo ""
srun $command

## Send email when job ends.
## echo " " | /usr/bin/mail -s "Sbatch $email_info ended" sfandres@unex.es
