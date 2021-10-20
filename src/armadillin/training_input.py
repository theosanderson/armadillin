from . import input
import pandas as pd
import numpy as np

metadata = pd.read_csv("/home/theo/sandslash/oct_metadata_cut.tsv",
                    sep="\t",
                    usecols=["strain", "gisaid_epi_isl"])

#metadata.to_csv("oct_metadata_cut.tsv", sep="\t", index=False)

epi_to_name = dict(zip(metadata["gisaid_epi_isl"], metadata["strain"]))

name_to_taxon = dict(zip(input.assignments['taxon'], input.assignments['lineage']))

epi_to_taxon = {
epi: name_to_taxon[name]
for epi, name in epi_to_name.items() if name in name_to_taxon
}

all_epis = list(epi_to_name.keys())

del metadata




import random


def random_sample_from_list(the_list, proportion):
    return random.sample(the_list, int(proportion * len(the_list)))


num_shards = 200
all_shards = range(num_shards)
train_shards = list(range(num_shards)) #random_sample_from_list(all_shards, 0.8)
test_shards = random_sample_from_list(all_shards, 0.1)

import gzip




def get_multi_hot_from_lineage(lineage):
    multi_hot = np.zeros(len(input.all_lineages), dtype=np.float32)

    while True:
        if lineage in input.lineage_to_index:
            multi_hot[input.lineage_to_index[lineage]] = 1
        if "." in lineage:
            subparts = lineage.split(".")
            lineage = ".".join(subparts[:-1])
            continue
        elif lineage in input.aliases and input.aliases[lineage] != "":
            lineage = input.aliases[lineage]
            continue
        else:
            assert lineage == "A" or lineage == "B"
            break
    return multi_hot



def yield_examples(shards, shard_dir):
    while True:
        for shard_num in shards:
            file = open(f"{shard_dir}/shard_{shard_num}.tsv")
            for line in file:
                epi, seq, lineage = line.strip().split("\t")
                if epi in epi_to_taxon:
                    lineage = epi_to_taxon[epi]
                    lineage_numpy = get_multi_hot_from_lineage_with_cache(
                        lineage)
                    #print(seq)
                    yield (input.string_to_one_hot_numpy(seq,target_length = 29891), lineage_numpy)


lineage_cache = {}


def get_multi_hot_from_lineage_with_cache(lineage):
    # check if lineage is a list, and raise error if so:

    if lineage in lineage_cache:
        return lineage_cache[lineage]
    lineage_numpy = get_multi_hot_from_lineage(lineage)
    lineage_cache[lineage] = lineage_numpy
    return lineage_numpy



dropout_prob = 0.1
#ref_dropout_prob = 0.05


def add_dropout(generator):
    while True:
        seq, lineage = next(generator)
        random_values = np.random.random((seq.shape[0], 1))
        seq = np.where(
            random_values > dropout_prob  #+ ref_dropout_prob
            ,
            seq,
            0)

        # seq = np.where(random_values < ref_dropout_prob, ref_numpy, seq)
        yield (seq, lineage)


def get_typed_examples(type, shard_dir):
    if type == "train":
        return add_dropout(yield_examples(train_shards, shard_dir))
    elif type == "test":
        return yield_examples(test_shards,shard_dir)


def yield_batch_of_examples(type, batch_size, shard_dir):
    example_iterator = get_typed_examples(type, shard_dir)
    while True:
        batch = [next(example_iterator) for _ in range(batch_size)]
        yield (np.stack([x[0] for x in batch]), np.stack([x[1]
                                                          for x in batch]))


