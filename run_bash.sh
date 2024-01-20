#!/bin/bash

#SBATCH --time=10:15:00
#SBATCH --job-name=Age
#SBATCH --output=output.txt
#SBATCH --error=error.txt

time python /cluster/home/lucabri/masterthesis/magging.py