# Armadillin

### This is an experimental tool under development. The recommended method for calling lineages remains normal Pangolin: https://github.com/cov-lineages/pangolin

_A Re-engineered Method Allowing DetermInation of viraL LINeages_

Armadillin is an experimental alternative approach to training models on [lineages designated by the PANGO team](https://github.com/cov-lineages/pango-designation).

Armadillin uses dense neural networks for assignment, which means it doesn't have to assume that positions with an N are the reference sequence. Armadillin is still very fast, in part because it sparsifies the feature input to this neural net during training.

## Installation (for inference)

```
conda create --name armadillin python=3.9
conda activate armadillin
pip3 install armadillin
```

## Usage

We'll use a COG-UK file for a demo:

```
wget https://cog-uk.s3.climb.ac.uk/phylogenetics/latest/cog_alignment.fasta.gz
```

```
armadillin https://cog-uk.s3.climb.ac.uk/phylogenetics/latest/cog_alignment.fasta.gz
```

or

```
armadillin https://cog-uk.s3.climb.ac.uk/phylogenetics/latest/cog_alignment.fasta.gz > output.tsv
```

## Training your own models

### Dataset generation

```
python -m armadillin.training_make_input --designations ~/gisaid/pango-designation-1.2.88/ --gisaid_meta_file ~/gisaid/metadata.tsv --gisaid_mmsa ~/gisaid/msa_2021-10-20.tar.xz --output ~/training_set_nov_02
 python -m armadillin.train --shard_dir /home/theo/training_set_nov_02 --use_wandb --checkpoint_path ~/nov2check1

 python -m armadillin.train --starting_model ~/nov2check1/checkpoint.h5 --use_wandb --checkpoint_path ~/nov2check1_sparse/ --do_pruning --shard_dir /home/theo/training_set_nov_02

 python -m armadillin.training_create_small_model -i /tmp/model_zeros.h5 -d  /home/theo/training_set_nov_02
```

## Related tools

[Pangolin](https://github.com/cov-lineages/pangolin) is the OG for assigning lineages
