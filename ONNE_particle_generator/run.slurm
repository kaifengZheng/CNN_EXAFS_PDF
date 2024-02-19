#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=167:59:59
#SBATCH --partition=extended-96core
#SBATCH --output=array_test.%N_%a.log
#SBATCH --job-name=EXAFS_1000_UF4
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

module load slurm
module load  python/3.9.7
module use modulefiles/FEFF
module load 10.0.0

#module load intel/oneAPI/2021.4

#module load  mpi/2021.4.0

##module load gnu-parallel

#rm -r output*&
#rm -r gr*&
#rm -r FEFF_inp&
#rm -r input
#wait

python run_FEFF.py -w
wait
python run_FEFF.py -r

#create particles
#python trainingset_gen_mpi_mygr.py -n 96
