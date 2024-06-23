#!/bin/bash -l


python init.py
sbatch cluster.sbatch 
echo "Job submitted. Good luck."