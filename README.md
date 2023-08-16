# pytorch-timing-example
This repository has been created for timing purposes of PyTorch scripts running on a cluster.

## Getting started

### Prerequisites
Anaconda distribution is recommended. You can install it following the [official installation guide](https://docs.anaconda.com/anaconda/install/linux/).

### Installation
The environment.yml file contains all the necessary packages to use this project inside the environment with name `pytorch-timing-conda` provided. You can create a conda environment from the [environment.yml](environment.yml) file provided as follows:
```
conda env create -f environment.yml
```

### Usage
The [timing_training_loop.py](timing_training_loop.py) measures the runtime of the training loop. It can be executed as follows:

* Under the Slurm cluster manager:
```
sbatch -p <partition_name> launch.sh -e 5 -b 64 -w 1
```
Remember to change the option `#SBATCH --cpus-per-task=X` in [launch.sh](launch.sh) according to the number of workers in the `-w` option of the script.

* Localhost:
```
conda activate pytorch-timing-conda
python3 timing_training_loop.py -e 5 -b 64 -w 4
```
