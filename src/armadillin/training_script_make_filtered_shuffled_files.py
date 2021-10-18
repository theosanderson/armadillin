import pandas as pd
import lzma

assignments = pd.read_csv("/home/theo/pango-designation/lineages.csv")

lineage_counts = assignments['lineage'].value_counts().to_dict()


metadata = pd.read_csv("/home/theo/sandslash/oct_metadata_cut.tsv",
                       sep="\t",
                       usecols=["strain", "gisaid_epi_isl"])

epi_to_name = dict(zip(metadata["gisaid_epi_isl"], metadata["strain"]))

name_to_taxon = dict(zip(assignments['taxon'], assignments['lineage']))

filename = "/home/theo/sandslash/oct_masked_alignment.tar.xz"
file = lzma.open(filename, "rt")
file.seek(1000)
cur_seq = ""
current_epi = ""

number_of_shards = 200
file_handles = {}
for i in range(number_of_shards):
    file_handles[i] = open(f"/home/theo/sandslash/shards/seq_shard_{i}.tsv", "wt")

import tqdm
import random

def write_to_random_handle(epi,seq
):
    random_handle = file_handles[random.randint(
        0, number_of_shards - 1)]
    random_handle.write(f"{epi}\t{seq}\n")

seq_is_good = False
import random
if True:
    with tqdm.tqdm(total=len(name_to_taxon)) as pbar:
        for line in file:
            if line.startswith(">"):
                if current_epi and seq_is_good:
                    lineage = name_to_taxon[epi_to_name[current_epi]]
                    if lineage_counts[lineage] >2000:
                        times_over_represented = lineage_counts[lineage] / 2000
                        if random.random() < (1/times_over_represented):
                            write_to_random_handle(current_epi, cur_seq)
                    elif lineage_counts[lineage] >1000:
                        write_to_random_handle(current_epi, cur_seq)
                    else:
                        times_to_write = int(500/lineage_counts[lineage])
                        times_to_write = min(times_to_write, 100)
                        for i in range(times_to_write):
                            write_to_random_handle(current_epi, cur_seq)

                current_epi = line.strip()[1:]
                if current_epi in epi_to_name and epi_to_name[
                        current_epi] in name_to_taxon:
                    seq_is_good = True
                    pbar.update(1)
                else:
                    seq_is_good = False
                cur_seq = ""
            else:
                if seq_is_good:
                    cur_seq += line.strip()

# Now close all files:
for i in range(number_of_shards):
    file_handles[i].close()

# Now open each in turn, shuffle the lines, and write out again:
for i in tqdm.tqdm(range(number_of_shards)):
    handle = open(f"/home/theo/sandslash/shards/seq_shard_{i}.tsv", "rt")
    lines = handle.readlines()
    random.shuffle(lines)
    handle.close()
    handle = open(f"/home/theo/sandslash/shards/seq_shard_{i}.tsv", "wt")
    handle.writelines(lines)
    handle.close()

import gzip

# Now merge all the files:
combined_file_name = "/home/theo/sandslash/shuffled_filtered_seqs.tsv.gz"
combined_file = gzip.open(combined_file_name, "wt")
for i in tqdm.tqdm(range(number_of_shards)):
    handle = open(f"/home/theo/sandslash/shards/seq_shard_{i}.tsv", "rt")
    combined_file.writelines(handle.readlines())
    handle.close()
