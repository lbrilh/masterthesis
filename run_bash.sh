#!/bin/bash

#SBATCH --time=01:15:00
#SBATCH --job-name=Numbeds
#SBATCH --output=output.txt
#SBATCH --error=error.txt

time python /cluster/home/lucabri/masterthesis/magging/experiments.py