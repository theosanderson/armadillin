import pandas as pd
import lzma
import numpy as np
from Bio import SeqIO
import os
import re
import unicodedata
import tarfile
import io
import gzip


def clean_string(text, strip_symbols=False):
    # strip accents
    text = ''.join((c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn'))
    # replace spaces with _
    text = text.replace(' ', '_')

    if strip_symbols:
        p = re.compile(r'[^a-zA-Z0-9_/-]')
        text = p.sub('', text)
    return text


def fix_header(header):
    """
    parse fasta header and remove problems
    """
    fixed_header = header.lstrip()\
        .replace(' ', '_')\
        .replace("hCoV-19/","")\
        .replace("hCov-19/","")\
        .replace("PENDING", "")\
        .replace("UPLOADED", "")\
        .replace("None", "")

    fixed_header = re.sub('^Wuhan-Hu-1', 'China/Wuhan-Hu-1', fixed_header)
    fixed_header = clean_string(fixed_header)

    return (fixed_header)

# Increase number of rows to print for pandas
pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 50)

#Create argparser
import argparse
parser = argparse.ArgumentParser(description='Create input for Armadillo training')
parser.add_argument('--designations', type=str, help='Path to pango-designation repository')
parser.add_argument('--output', type=str, help='Path to output directory')
parser.add_argument('--gisaid_meta_file', type=str, help='Path to GISAID metadata file')
parser.add_argument('--gisaid_mmsa', type=str, help='Path to GISAID MSA')


args = parser.parse_args()

#If output dir does not exist, create it:
if not os.path.exists(args.output):
    os.makedirs(args.output)


assignments = pd.read_csv(f"{args.designations}/lineages.csv")
all_lineages = sorted(list(set(assignments['lineage'])))
# save all_lineages to file
with open(f"{args.output}/all_lineages.txt", "w") as f:
    for l in all_lineages:
        f.write(f"{l}\n")

# read entire contents of aliases file
aliases_file = open(f"{args.designations}/pango_designation/alias_key.json", "rt")

# save contents of aliases to output
with open(f"{args.output}/aliases.json", "w") as f:
    for line in aliases_file:
        f.write(line)



total_sampling = 1

lineage_counts = assignments['lineage'].value_counts().to_frame()

print(f"AY.4 starting count is {lineage_counts['lineage']['AY.4']}")
print(f"AY.6 starting count is {lineage_counts['lineage']['AY.6']}")

print(f"B.1.1.7 starting count is {lineage_counts['lineage']['B.1.1.7']}")
print(f"Q.1 starting count is {lineage_counts['lineage']['Q.1']}")


lineage_counts['nonlinear'] =lineage_counts['lineage']**.22
lineage_counts['nonlinear_ratio'] =lineage_counts['nonlinear'].sum()/ lineage_counts['nonlinear'] 
lineage_counts['nonlinear_ratio'] = lineage_counts['nonlinear_ratio']  / 500
lineage_counts['expectation'] = lineage_counts['lineage'] * lineage_counts['nonlinear_ratio']



print(f"Expecting total size of: {lineage_counts['expectation'].sum()}")
print(f"Expecting this many Q.1: {lineage_counts['expectation']['Q.1']}")
print(f"Expecting this many B.1.1.7: {lineage_counts['expectation']['B.1.1.7']}")




metadata = pd.read_csv(args.gisaid_meta_file,
                       sep="\t",
                       usecols=["Virus name", "Accession ID"])

# rename Virus name to strain and Accession ID to gisaid_epi_isl
metadata = metadata.rename(columns={"Virus name": "strain", "Accession ID": "gisaid_epi_isl"})
#Replace spaces with underscores in strain
metadata['strain'] = metadata['strain'].apply(fix_header)

epi_to_name = dict(zip(metadata["gisaid_epi_isl"], metadata["strain"]))

name_to_lineage = dict(zip(assignments['taxon'], assignments['lineage']))
#print(name_o_taxon)

# Get list of names that are "AY.6"
AY_6_names = assignments[assignments['lineage'] == 'AY.6']['taxon'].tolist()

#raise ValueError(AY_6_names)
# Get epis from these names
AY_6_epis = metadata[metadata['strain'].isin(AY_6_names)]['gisaid_epi_isl'].tolist()



#raise ValueError((AY_6_epis))

filename = args.gisaid_mmsa



if filename.endswith("tar.xz"):
            tar = tarfile.open(filename, "r:xz")
            while True:
                member = tar.next()
                if member.size>100000:
                    break
            if member is None:
                raise ValueError("Could not find member")
            
            handle = tar.extractfile(member)
            handle = io.TextIOWrapper(handle) 
            print("Using largest file as handle")
            print("Using tarxz mode")
            
else:
    handle = open(filename,"rt")

number_of_shards = 400
file_handles = {}

def get_shard_path(i):
    return f"{args.output}/shard_{i}.tsv.gz"
for i in range(number_of_shards):
    file_handles[i] = gzip.open(get_shard_path(i), "wt")

import tqdm
import random

metadata_num_rows = metadata.shape[0]

def write_to_random_handle(epi,seq
,lineage):
    #print("writing")
    random_handle = file_handles[random.randint(
        0, number_of_shards - 1)]
    random_handle.write(f"{epi}\t{seq}\t{lineage}\n")

seq_is_good = False

total = 0
import random
if True:
    with tqdm.tqdm(total=metadata_num_rows) as pbar:
        for record in SeqIO.parse(handle, "fasta"):
            pbar.update(1)
            total +=1
            #if total>10000:
            #    break
            epi = record.id
            seq = str(record.seq)
            try:
                lineage = name_to_lineage[epi_to_name[epi]]
            except KeyError:
                if epi not in epi_to_name:
                    print(f"{epi} not in epi_to_name")
                else:
                   #print(f"{epi_to_name[epi]} not in name_to_lineage")
                   pass
                continue
            
            times_to_sample = lineage_counts['nonlinear_ratio'][lineage]
            #print(f"{lineage} is {times_to_sample} times nonlinear")
            if times_to_sample > 1:
                for i in range(int(times_to_sample)):
                    write_to_random_handle(epi, seq, lineage)
            else:
                rand_no = random.random()
                if  rand_no< times_to_sample:
                     write_to_random_handle(epi, seq, lineage)

print("Completed, closing all files")
# Now close all files:
for i in range(number_of_shards):
    file_handles[i].close()

print("Beginning shuffle")
# Now open each in turn, shuffle the lines, and write out again:
for i in tqdm.tqdm(range(number_of_shards)):
    print("opening this file")
    handle = gzip.open(get_shard_path(i), "rt")
    
    lines = handle.readlines()
    random.shuffle(lines)
    handle.close()
    print("loaded this file")
    handle = gzip.open(get_shard_path(i), "wt")
    handle.writelines(lines)
    handle.close()


print("Shuffle complete")
