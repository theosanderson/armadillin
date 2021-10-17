# Armadillin
_A Re-engineered Method Allowing DetermInation of viraL LINeages_

Armadillin is an experimental alternative approach that trains models on [lineages designated by the PANGO team](https://github.com/cov-lineages/pango-designation).

Armadillin uses dense neural networks for assignment, which means it doesn't have to assume that positions with an N are the reference sequence.

## Installation (for inference)
```
conda create --name armadillin python=3.9
conda activate armadillin
pip install armadillin
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
While training code is in the repo it is not quite in working order today and needs a little work.

## Related tools
[Pangolin](https://github.com/cov-lineages/pangolin) is the OG for assigning lineages
