import numpy as np
import pandas as pd
from Bio import SeqIO
from . import cov2_genome
import pkg_resources
import json
alphabet = "acgt-"
import gzip

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


def string_to_one_hot_numpy(string, target_length = None):
    if target_length:
        if len(string)> target_length:
            string = string[0:target_length]
        if len(string) < target_length:
            # pad string to desired length
            string = string + "n" * (target_length - len(string))
        
    as_numbers = string_to_ints(string)

    return character_lookup_table[as_numbers]




ref_numpy = string_to_one_hot_numpy(cov2_genome.seq.lower()[0:29891])



assignments = pd.read_csv(  pkg_resources.resource_stream(__name__, 'trained_model/lineages.csv'))




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





lineage_to_index = dict(zip(all_lineages, range(len(all_lineages))))


def yield_from_fasta(filename, mask=None):
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
        yield seq_id, string_to_one_hot_numpy(seq).flatten()


def apply_mask_to_numpy_iterator(seq_iterator, selected_indices):
    for seq_id, flat in seq_iterator:
        masked_flat = flat[selected_indices]
        
        yield seq_id, masked_flat
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
