#!/bin/bash

#SBATCH --time=10:15:00
#SBATCH --job-name=Numbeds
#SBATCH --output=output.txt
#SBATCH --error=error.txt

time python '/cluster/home/lucabri/masterthesis//magging/nonlinear version/nonlinear_magging.py'