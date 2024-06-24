# CNN_EXAFS_PDF
This repo contains all codes related to the work entitled "Decoding the Pair Distribution Function of Uranium in Molten Fluoride Salts from   X-ray Absorption Spectroscopy Data by Machine Learning" [https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.4c01898](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.4c01898)
# description of files
```markdown
CNN_EXAFS_PDF
| CNN sweep_WB.ipynb
      script to tune parameters of neural network using W&B
| NN_300_partial_full_multi_final.ipynb
      main notebook to execute neural network to predict g(r) and validate the results using experimental data
|__MD_data
   | gr_output*.csv
        PDFs from MD frames
   | output_ave_test_kx_MD.csv
        EXAFS of MD frames
   | r_*.txt
        rmesh
|__ONNE_particle_generator
   | gr.csv
   | gr.png
   | config.toml
        configuration file for ONNE particle data construction
   | rmesh.txt
   | run.slurm
       script to run on supercluster
   | trainingset_gen_mpi_mygr.py
       main script to run ONNE particle construction
   |__Scripts
       contains util scripts for generating particles and calculate EXAFS
|__config
   | config_1116023.yaml
|__cbam/rebinE0/lightning_logs
      checkpoints of neural network model
   |__version_*
      |__checkpoints
|__multi_tasks_FFFF
     FEFF calculation using multi-processing
   | config_wr.toml
       configuration file for writing FEFF inputs from xyz files and run FEFF calculations
   | output_ave_test_kx.csv
   | run.slurm
   | run_FEFF.py
       main script for run FEFF and write FEFF input files
   | template.inp
   |__toolbox
      | average.py
         calculate average EXAFS for each case
      | average_gr.py
         calculate average gr for each case
      | check.py
         check if the calculations are completed
      | run_tool.sh
      | srun.sh
|__dataset_partial
   contains gr/EXAFS data for ONNE training purpose
   | gr_*.csv
   | gr.png
   | kmesh.txt
   | rmesh.txt
   | output_ave_test_kx.csv
   | output_rebin.csv

```
Contributors:
- multi_tasks_FEFF:
  - [@Kaifeng Zheng](https://github.com/kaifengZheng)
  - [@Mehmet Toposakal](https://github.com/MehmetTopsakal)
- ONNE_particle_generator:
  - [@Kaifeng Zheng](https://github.com/kaifengZheng)
  - [@Nicholas Marcella](https://github.com/nmarcella)
- neural network construction:
  - [@Kaifeng Zheng](https://github.com/kaifengZheng)
    - combinatorial method
      -[@Nicholas Marcella](https://github.com/nmarcella)
