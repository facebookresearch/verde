This repository contains code to recreate the results from [*SALSA VERDE: a machine learning attack on Learning With Errors with sparse small secrets*](https://arxiv.org/abs/2306.11641), which uses transformers to recover secrets from LWE samples ($\mathbf{a}$, $b$). The code in this repo can also be used to run the attack in [*SALSA PICANTE: a Machine Learning Attack on LWE with Binary Secrets*](https://arxiv.org/abs/2303.04178). The Verde attack strictly supersedes the Picante attack in terms of performance. 

## Quickstart

__Installation__: To get started, clone the repository to a machine that has at least one gpu. Create the necessary conda environment via ```conda create --name lattice_env --file requirements.txt``` and activate your shiny new environment via ```conda activate lattice_env```.

__Download data__: For ease of use, we have provided a pre-processed dataset for you to use. It will enable you to run experiments on $n=256$, $log_2 q=20$ data with sparse binary secrets. You can download the data from [this link](https://dl.fbaipublicfiles.com/verde/n256_logq20_binary_for_release.tar.gz). The data folder contains the following files: 
  - orig_A.npy and orig_b.npy: the original 4n LWE samples
  - params.pkl: the params for the dataset, the transformer reads this file so it isn't necessary to specify lwe parameters (N, Q, sigma) for this run. 
  - secret.npy: a file containing various secrets
  - test.prefix and train.prefix: the preprocessed test and train sets, each line is a string of {a} ; {b}

__Your first experiment__: Once you've done this, run ```python3 train.py --reload_data /path/to/data --secret_seed 3 --hamming 30 --input_int_base 105348 --share_token 64 --optimizer adam_warmup,lr=0.00001,warmup_updates=1000,warmup_init_lr=0.00000001```. This will train a model on the preprocessed dataset ($n=256$, $log_2q=20$, $h=30$). The input encoding base and share token for this setting are specified in Table 9 in VERDE's Appendix A.1, and the model architecture is specified in Section 2 of the paper. This model runs smoothly on a single NVIDIA Quadro GV100 32GB. It should take roughly ~2 hours per epoch to run, and, if a secret is recovered, this should happen in early epochs. You can re-run the experiment with a different secret seed (range is 0-9) or Hamming weight (range is 3-40) if this experiment fails -- remember that not all attacks succeed on the first try!

__Parameters you can play with__: 
Although you can vary the parameters as you see fit, the default training parameters are specified as defaults in ```train.py``` and the ```params.pkl``` file provided with the dataset. Note that this codebase currently only supports the seq2seq model, not the encoder-only model tested in Section 7 of the paper. 
- Model architecture parameters (defined in ```src/train.py```):
  - ```enc_emb_dim```: encoder's embedding dimension
  - ```dec_emb_dim```: decoder's embedding dimension
  - ```n_enc_layers```: number of layers in encoder
  - ```n_dec_layers```: number of layers in decoder
  - ```n_enc_heads```: number of attention heads in encoder
  - ```n_dec_heads```: number of attention heads in decoder
  - ```enc_loops```: number of loops through encoder (Universal Transformer parameter)
  - ```dec_loops```: number of loops through decoder (Universal Transformer parameter)
- Training parameters
  - ```epoch_size```: number of LWE samples per epoch
  - ```batch_size```: how many LWE samples per batch
- LWE problem parameters
  - ```N```: lattice dimension
  - ```Q```: prime modulus for LWE problem
  - ```sigma```: stdev of error distribution used in LWE
  - ```hamming```: Hamming weight of binary LWE secret
  - ```input_int_base```: integer encoding base for transformer inputs
  - ```output_int_base```: integer encoding base for transformer outputs

__Running sweeps with slurm__: To run sweeps on our cluster, we use slurm to parse the json files and farm out experiments to machines. If you add additional elements to the lists in the json files (e.g. ```hamming: [30, 35]``` instead of just ```hamming: [30]```) and use an appropriate parser (e.g. ), you too can run sweeps locally. 

__Analyzing results__: If you have a large set of experiments you want to analyze, you can use ```./notebooks/LatticeMLReader.ipynb```. This will parse log file(s) from a given experiment(s) and provides other helpful information.

__Generating your own data__: If you are interested in generating your own reduced data to run a different attack, proceed as follows.
  - First, generate the a vectors of the original lwe samples with all entries as integers [0, q). Store it as a 4n x n matrix in a numpy file (see the orig_A.npy in the provided data folder as an example). 
  - Then, run the lattice reduction preprocessing with the following command: ```python generate.py --timeout 432000 --N 256 --Q 842779 --lll_delta 0.99 --float_type dd --bkz_block_size 35 --threshold 0.435 --threshold2 0.5 –use_polish true --step RA_tiny2 --reload_data <path of orig_A.npy>``` (you can change N, Q, etc., depending on the attack you want to run).
  - This creates a directory for the reduced matrices. Inside the directory, there should be a params.pkl and data.prefix, where the reduced matrices are written into. 
  - The last step is to get a dataset of reduced (A, b) with the following command:
  ```python generate.py --min_hamming 16 --max_hamming 25 --num_workers 1 --num_secret_seeds 5 --step Ab --secret_type binary --epoch_size 1000000 --reload_size 1000000 --reload_data <path of directory for the reduced matrices>```. This will create the dataset with 50 binary secrets with Hamming weight ranging from 16 to 25, 5 at each Hamming weight. 

Now you have a set of reduced matrices on which you can run attacks! The command provided above for training models on the provided data should also work on this dataset, as long as you change the path to point at your own reduced data. 

If you want to run the two preprocessing steps above using slurm, we have provided two .json files in the ```slurm_params``` folder: ```create_n256_data_step1.json``` and ```create_n256_data_step2.json```. These files provide helpful examples for setting up sbatch (or similar slurm scheduling tool) runs. 

## Citing this repo

Please use the following citation for this repository. 

```
@inproceedings{li2023salsa,
  title={SALSA VERDE: a machine learning attack on Learning With Errors with sparse small secrets},
  author={Li, Cathy and Wenger, Emily and Allen-Zhu, Zeyuan and Charton, Francois and Lauter, Kristin},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2023}
}
```

## License - Community

SALSA VERDE is licensed, as per the license found in the LICENSE file.
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.
