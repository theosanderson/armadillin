import pandas as pd
import lzma
import numpy as np

# Increase number of rows to print for pandas
pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 50)
assignments = pd.read_csv("/home/theo/pango-designation/lineages.csv")

total_sampling = 1

lineage_counts = assignments['lineage'].value_counts().to_frame()

print(f"AY.4 starting count is {lineage_counts['lineage']['AY.4']}")
print(f"AY.6 starting count is {lineage_counts['lineage']['AY.6']}")


lineage_counts['nonlinear'] =lineage_counts['lineage']**.88
lineage_counts['nonlinear_ratio'] =lineage_counts['nonlinear'].sum()/ lineage_counts['nonlinear'] 
lineage_counts['nonlinear_ratio'] = lineage_counts['nonlinear_ratio']  / 1000
lineage_counts['expectation'] = lineage_counts['lineage'] * lineage_counts['nonlinear_ratio']


print(lineage_counts)
print(lineage_counts['expectation'].sum())
print(lineage_counts['nonlinear_ratio']['AY.4'])
print(lineage_counts['nonlinear_ratio']['AY.6'])

print(f"AY.4 expected count is {lineage_counts['expectation']['AY.4']}")
print(f"AY.6 expected count is {lineage_counts['expectation']['AY.6']}")



metadata = pd.read_csv("/home/theo/sandslash/oct_metadata_cut.tsv",
                       sep="\t",
                       usecols=["strain", "gisaid_epi_isl"])

epi_to_name = dict(zip(metadata["gisaid_epi_isl"], metadata["strain"]))

name_to_taxon = dict(zip(assignments['taxon'], assignments['lineage']))

# Get list of names that are "AY.6"
AY_6_names = assignments[assignments['lineage'] == 'AY.6']['taxon'].tolist()

#raise ValueError(AY_6_names)
# Get epis from these names
AY_6_epis = metadata[metadata['strain'].isin(AY_6_names)]['gisaid_epi_isl'].tolist()

#raise ValueError((AY_6_epis))

filename = "/home/theo/mmsa_2021-10-15.tar.xz"
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
,lineage):
    random_handle = file_handles[random.randint(
        0, number_of_shards - 1)]
    random_handle.write(f"{epi}\t{seq}\t{lineage}\n")

seq_is_good = False
import random
if True:
    with tqdm.tqdm(total=len(name_to_taxon)) as pbar:
        for line in file:
            if line.startswith(">"):
                if current_epi and seq_is_good:
                    lineage = name_to_taxon[epi_to_name[current_epi]]
                    if(lineage == 'AY.6'):
                        print("Found an AY.6")
                    times_to_sample = lineage_counts['nonlinear_ratio'][lineage]
                    
                    if times_to_sample > 1:
                        if lineage=="AY.6":
                            print(f"Writing {current_epi} of lineage {lineage}  {times_to_sample}  times")
                        for i in range(int(times_to_sample)):
                            write_to_random_handle(current_epi, cur_seq, lineage)
                    else:
                        rand_no = random.random()
                        if  rand_no< times_to_sample:
                            if lineage=="AY.6":
                                print(f"Writing {current_epi} of lineage {lineage} because {rand_no} was less than {times_to_sample}")
                            write_to_random_handle(current_epi, cur_seq, lineage)

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
