# Running Jobs

## Setup

To run a job, create and activate a Conda environment, then install the required dependencies for the task you want to run.

- **gpu_mode** Conda environments must use **Python 3.13.11**
- All other tasks are recommended to use **Python 3.11**

Each task has its own `requirements.txt` located under:

requirements/

Install dependencies with:

pip install -r requirements/<task_name>.txt

## Running Tasks

After installing dependencies, locate the corresponding task script under:

scripts/

Each task provides a Bash script that launches the job. Run it with:

bash scripts/<task_name>/run.sh

## Multi-node Execution

Multi-node execution is supported via Slurm.

## Security Notice

It is **highly recommended** to run all jobs on an isolated network or VPN. Ray has minimal built-in security protections and should not be exposed on a public or shared network.

## Hardware Requirements and Performance Notes

All reported results were run using HPC-grade CPUs.

Mathematics and AHC tasks will perform significantly worse if they are not run on HPC-grade CPUs or if they are limited to a small number of cores. For these tasks, it is strongly recommended to use a large number of CPU cores and multiple hosts.

## AHC Container Requirements

For AHC tasks, jobs must be launched inside the ALE-Bench provided C++ container:

yimjk/ale-bench:cpp20-202301

Docker Hub:
https://hub.docker.com/layers/yimjk/ale-bench/cpp20-202301/images/sha256-946af1b209a84160594e43262b5157aec933938c99e2585b53042cac9bc3f43c

We support the Pyxis Slurm plugin to launch this container across multiple nodes for AHC, but it is not strictly required.