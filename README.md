# CNN_EXAFS_PDF
This repo contains all codes related to the work entitled "Decoding the Pair Distribution Function of Uranium in Molten Fluoride Salts from   X-ray Absorption Spectroscopy Data by Machine Learning" [https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.4c01898](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.4c01898)
# description of files
```markdown
CNN_EXAFS_PDF
| CNN sweep_WB.ipynb
      script for tuning hyperparameters of our neural network using W&B
| NN_300_partial_full_multi_final.ipynb
      Main notebook to execute a neural network for predicting g(r) and validating the results using experimental data
|__MD_data
   | gr_output*.csv
        PDFs calculated from MD structures
   | output_ave_test_kx_MD.csv
        FEFF-EXAFS calculated using MD structures 
   | r_*.txt
        rmesh
|__ONNE_particle_generator
   | gr.csv
   | gr.png
   | config.toml
        configuration file for ONNE particle data construction
   | rmesh.txt
   | run.slurm
       script to run the simulation/particle generator on supercluster
   | trainingset_gen_mpi_mygr.py
       main script to run ONNE particle construction
   |__Scripts
       Contains utility scripts for generating particles and calculating EXAFS
|__config
   | config_1116023.yaml
     trained parameters for the neural networks.
|__cbam/rebinE0/lightning_logs
      checkpoints of neural network model
   |__version_*
      |__checkpoints
|__multi_tasks_FFFF
     FEFF calculation using multi-processing
   | config_wr.toml
       configuration file for writing FEFF input files from xyz files and running FEFF calculations
   | output_ave_test_kx.csv
   | run.slurm
   | run_FEFF.py
       main script for run FEFF and write FEFF input files
   | template.inp
   |__toolbox
      | average.py
         calculate average EXAFS for all gr configurations
      | average_gr.py
         calculate averaged gr for all gr configurations
      | check.py
         Check if the calculations are complete
      | run_tool.sh
      | srun.sh
|__dataset_partial
   gr/EXAFS training data
   | gr_*.csv
   | gr.png
   | kmesh.txt
   | rmesh.txt
   | output_ave_test_kx.csv
   | output_rebin.csv

```
Contributors:
- multi_tasks_FEFF(also see [project](https://github.com/kaifengZheng/FEFF_package.git)):
  - [@Kaifeng Zheng](https://github.com/kaifengZheng)
  - [@Mehmet Toposakal](https://github.com/MehmetTopsakal)
- ONNE_particle_generator(also see the original repo: credits to [@Nicholas Marcella Xron repo](https://github.com/nmarcella/Xron)):
  - [@Kaifeng Zheng](https://github.com/kaifengZheng)
- neural network construction:
  - [@Kaifeng Zheng](https://github.com/kaifengZheng)
    - combinatorial method
      - [@Nicholas Marcella](https://github.com/nmarcella)
