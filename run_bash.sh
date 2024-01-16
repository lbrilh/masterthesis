#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --job-name=Magging 
#SBATCH --output=output.txt
#SBATCH --error=error.txt

python /cluster/home/lucabri/masterthesis/magging.py