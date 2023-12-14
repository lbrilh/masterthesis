#!/bin/bash

#SBATCH --time=00:08:00
#SBATCH --job-name=LGBMRefit 
#SBATCH --output=output.txt
#SBATCH --error=error.txt

python /cluster/home/lucabri/masterthesis/experiments_script.py