#!/bin/bash

#SBATCH --time=10:01:00
#SBATCH --job-name=HospitalID
#SBATCH --output=output.txt
#SBATCH --error=error.txt

time python /cluster/home/lucabri/masterthesis/magging.py