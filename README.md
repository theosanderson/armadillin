# Armadillin

# DO NOT USE WITH CURRENT MODELS.


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
You must already have aligned your files to the reference (doing this automatically is on the backlist).

We'll use the COG-UK aligned file for a demo:
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
While training code is in the repo, it is not quite in easy working order today and needs a little work.

## Related tools
[Pangolin](https://github.com/cov-lineages/pangolin) is the OG for assigning lineages
