# Jobscript for pyRadiomics 

#$ -S /bin/bash

# to request x minutes: (calculate with 10 min / acquisition for all 541 patients 
#$ -l h_rt=1:00:0 

# request amount of memory. values must be equal. before full dataset, benchmark requirements. 
#$ -l tmem=3G
 #$ -l h_vmem=3G

# working directory
#$ -wd /home/goliveir

#joins output two output files into one
#$ -j y

# provide a name for your job
#$ -N pyRadiomics

# Commands to be executed go here:
# /home/goliveir 
date 
hostname

# prepare environment
source /share/apps/source_files/python/python-3.7.0.source
export PYRADIOMICS_ENV=cluster
python3 /home/goliveir/pyRadiomics/Notebook/feature_extraction_cluster.py "Exponential" # Choose from: ["Original", "SquareRoot", "Wavelet", "Square", "Exponential", "Logarithm"]

date
