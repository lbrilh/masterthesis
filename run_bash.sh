#!/bin/bash

#SBATCH --time=10:01:00
#SBATCH --job-name=Services
#SBATCH --output=output.txt
#SBATCH --error=error.txt

time python /cluster/home/lucabri/masterthesis/magging.py