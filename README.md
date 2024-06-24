# CNN_EXAFS_PDF
This repo contains all codes related to the work entitled "Decoding the Pair Distribution Function of Uranium in Molten Fluoride Salts from   X-ray Absorption Spectroscopy Data by Machine Learning" [https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.4c01898](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.4c01898)
# description of files
```
CNN_EXAFS_PDF
| CNN sweep_WB.ipynb
| NN_300_partial_full_multi_final.ipynb
|__MD_data
   | gr_UF4*.csv
   | gr_output*.csv
   | output_ave_test_kx_MD.csv
   | r_*.txt
|__ONNE_particle_generator
   | gr.csv
   | gr.png
   | config.toml
   | config_wr.toml
   | output.dat
   | output_ave_test_kx.csv
   | rmesh.txt
   | run.slurm
   | run_FEFF.py
   | template.inp
   | trainingset_gen_mpi_mygr.py
   |__Script
   |__toolbox
|__config
   |config_1116023.yaml
|__cbam/rebinE0/lightning_logs
   |__version_*
      |__checkpoints
|__dataset_partial
   |gr_*.csv
   |gr.png
   |kmesh.txt
   |rmesh.txt
   |output_ave_test_kx.csv
   |output_rebin.csv

```
