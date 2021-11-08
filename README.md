# Armadillin

### This is an experimental tool under development. The recommended method for calling lineages remains normal Pangolin: https://github.com/cov-lineages/pangolin

_A Re-engineered Method Allowing DetermInation of viraL LINeages_

Armadillin is an experimental alternative approach to training models on [lineages designated by the PANGO team](https://github.com/cov-lineages/pango-designation).

Armadillin uses dense neural networks for assignment, which means it doesn't have to assume that positions with an N are the reference sequence. Armadillin is still very fast, in part because it sparsifies the feature input to this neural net during training.

## Installation

### With pipx (self-contained)
```
pip install --local pipx
pipx install  armadillin
```

### In your environment
```
pip3 install armadillin
```

## Usage

We'll use a NextStrain open sampled file for a demo:

```
wget https://data.nextstrain.org/files/ncov/open/global/sequences.fasta.xz
```

```
armadillin sequences.fasta.xz
```

or

```
armadillin sequences.fasta.xz > output.tsv
```

If you have sequences already aligned to the reference you can make inference much faster using the `--seqs_are_aligned` parameter: 

```
wget https://data.nextstrain.org/files/ncov/open/global/aligned.fasta.xz
armadillin aligned.fasta.xz --seqs_are_aligned > output.tsv
```

## Related tools

[Pangolin](https://github.com/cov-lineages/pangolin) is the OG for assigning lineages
