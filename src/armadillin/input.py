import numpy as np
import pandas as pd
from Bio import SeqIO
from . import cov2_genome
import pkg_resources
import json
alphabet = "acgt-"

from collections import defaultdict


def string_to_ints(string):
    as_array = np.array((string), "c")
    as_numbers = as_array.view(np.int8)
    return as_numbers


def make_character_lookup_table(alphabet):
    character_lookup_table = np.zeros((256, len(alphabet)))
    indices = string_to_ints(alphabet)
    for i, x in enumerate(indices):
        character_lookup_table[x][i] = 1

    indices = string_to_ints(alphabet.upper())
    for i, x in enumerate(indices):
        character_lookup_table[x][i] = 1
    return character_lookup_table


character_lookup_table = make_character_lookup_table(alphabet)


def string_to_one_hot_numpy(string):
    as_numbers = string_to_ints(string)

    return character_lookup_table[as_numbers]




ref_numpy = string_to_one_hot_numpy(cov2_genome.seq.lower()[0:29891])



assignments = pd.read_csv(  pkg_resources.resource_stream(__name__, 'trained_model/lineages.csv'))


# Have removed the stuff below as it is only needed for training, todo refactor:
IS_TRAINING = False
if IS_TRAINING:
    metadata = pd.read_csv("oct_metadata_cut.tsv",
                        sep="\t",
                        usecols=["strain", "gisaid_epi_isl"])

    #metadata.to_csv("oct_metadata_cut.tsv", sep="\t", index=False)

    epi_to_name = dict(zip(metadata["gisaid_epi_isl"], metadata["strain"]))

    name_to_taxon = dict(zip(assignments['taxon'], assignments['lineage']))

    epi_to_taxon = {
    epi: name_to_taxon[name]
    for epi, name in epi_to_name.items() if name in name_to_taxon
    }

    all_epis = list(epi_to_name.keys())

    del metadata

aliases = json.load(

    pkg_resources.resource_stream(__name__, 'trained_model/alias_key.json')
    
    )

aliases['XA'] = "B.1"


def get_unaliased_lineage(lineage):
    components = lineage.split(".")
    if components[0] in aliases:
        return aliases[components[0]] + "." + ".".join(components[1:])


all_lineages = sorted(list(set(assignments['lineage'])))

all_lineages_df = pd.DataFrame({'lineage': all_lineages})
all_lineages_df['dealiased'] = all_lineages_df['lineage'].apply(
    get_unaliased_lineage)
all_lineages_df['level'] = all_lineages_df['dealiased'].str.count("\.")

lineage_to_level = dict(
    zip(all_lineages_df['lineage'], all_lineages_df['level']))




def get_multi_hot_from_lineage(lineage):
    multi_hot = np.zeros(len(all_lineages), dtype=np.float32)

    while True:
        if lineage in lineage_to_index:
            multi_hot[lineage_to_index[lineage]] = 1
        if "." in lineage:
            subparts = lineage.split(".")
            lineage = ".".join(subparts[:-1])
            continue
        elif lineage in aliases and aliases[lineage] != "":
            lineage = aliases[lineage]
            continue
        else:
            assert lineage == "A" or lineage == "B"
            break
    return multi_hot




lineage_to_index = dict(zip(all_lineages, range(len(all_lineages))))


import random


def random_sample_from_list(the_list, proportion):
    return random.sample(the_list, int(proportion * len(the_list)))


num_shards = 200
all_shards = range(num_shards)
train_shards = random_sample_from_list(all_shards, 0.8)
test_shards = list(set(all_shards) - set(train_shards))

import gzip


def yield_examples(shards):
    while True:
        for shard_num in shards:
            file = open(f"shards/seq_shard_{shard_num}.tsv")
            for line in file:
                epi, seq = line.strip().split("\t")
                if epi in epi_to_taxon:
                    lineage = epi_to_taxon[epi]
                    lineage_numpy = get_multi_hot_from_lineage_with_cache(
                        lineage)
                    #print(seq)
                    yield (string_to_one_hot_numpy(seq), lineage_numpy)


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


def get_typed_examples(type):
    if type == "train":
        return add_dropout(yield_examples(train_shards))
        return yield_examples(train_shards)
    elif type == "test":
        return yield_examples(test_shards)


def yield_batch_of_examples(type, batch_size):
    example_iterator = get_typed_examples(type)
    while True:
        batch = [next(example_iterator) for _ in range(batch_size)]
        yield (np.stack([x[0] for x in batch]), np.stack([x[1]
                                                          for x in batch]))


def batch_singles(iterator, batch_size):
    batch = []
    while True:
        try:
            batch.append(next(iterator))
            if len(batch) == batch_size:
                yield np.stack(batch)
                batch = []
        except StopIteration:
            if len(batch) > 0:
                yield np.stack(batch)
            return


def yield_from_fasta(filename="./cog_alignment.fa", mask=None):
    if filename.endswith(".gz"):
        handle = gzip.open(filename, "rt")
    else:
        handle = open(filename, "rt")
    for record in SeqIO.parse(handle, "fasta"):
        if mask != None:
            raise ValueError
        else:
            yield record.id, str(record.seq)[0:29891]


def apply_mask_to_seq_iterator(seq_iterator, selected_indices):
    for seq_id, seq in seq_iterator:
        masked_seq = ""
        for i in selected_indices:
            masked_seq += seq[i]
        yield seq_id, masked_seq


def apply_numpy_to_seq_iterator(seq_iterator):
    for seq_id, seq in seq_iterator:
        yield seq_id, string_to_one_hot_numpy(seq)
