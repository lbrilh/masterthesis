#!/bin/bash

#SBATCH --time=10:05:00
#SBATCH --job-name=Region
#SBATCH --output=output.txt
#SBATCH --error=error.txt

time python /cluster/home/lucabri/masterthesis/magging.py