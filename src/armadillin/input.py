import numpy as np
import pandas as pd
from Bio import SeqIO
from . import cov2_genome
import pkg_resources
import json
import lzma
import zipfile
import tarfile
alphabet = "acgt-"
import gzip
import os
import io

from collections import defaultdict



class Input(object):
    def __init__(self, path):
        self.path = path
        self.aliases = json.load(open(os.path.join(path, "aliases.json")))
        self.character_lookup_table = self.make_character_lookup_table(alphabet)
        self.ref_numpy = self.string_to_one_hot_numpy(cov2_genome.seq.lower()[0:29891])
        self.all_lineages = [x.strip() for x in open(os.path.join(path, "all_lineages.txt"))]
        self.lineage_to_index = {x:i for i, x in enumerate(self.all_lineages)}
        self.aliases['XA'] = 'B.1'
        self.lineage_to_level = {x:self.get_unaliased_lineage(x).count(".") for x in self.all_lineages}

    def string_to_ints(self, string):
        as_array = np.array((string), "c")
        as_numbers = as_array.view(np.int8)
        return as_numbers


    def make_character_lookup_table(self, alphabet):
        character_lookup_table = np.zeros((256, len(alphabet)))
        indices = self.string_to_ints(alphabet)
        for i, x in enumerate(indices):
            character_lookup_table[x][i] = 1

        indices = self.string_to_ints(alphabet.upper())
        for i, x in enumerate(indices):
            character_lookup_table[x][i] = 1
        return character_lookup_table


    


    def string_to_one_hot_numpy(self, string, target_length = None):
        if target_length:
            if len(string)> target_length:
                string = string[0:target_length]
            if len(string) < target_length:
                # pad string to desired length
                string = string + "n" * (target_length - len(string))
            
        as_numbers = self.string_to_ints(string)

        return self.character_lookup_table[as_numbers]




    





    def get_unaliased_lineage(self, lineage):
        components = lineage.split(".")
        if components[0] in self.aliases:
            return self.aliases[components[0]] + "." + ".".join(components[1:])


    def yield_from_fasta(self, filename, mask=None):
        if filename.endswith(".gz"):
            handle = gzip.open(filename, "rt")
        elif filename.endswith("tar.xz"):
            tar = tarfile.open(filename, "r:xz")
            members = tar.getmembers()
            # find largest member:
            largest_member = max(members, key=lambda x: x.size)
            handle = tar.extractfile(largest_member)
            handle = io.TextIOWrapper(handle)
        elif filename.endswith(".xz"):
            handle = lzma.open(filename, "rt")
        elif filename.endswith(".zip"):
            print(f"Opening {filename} as zip")
            the_zip = zipfile.ZipFile(filename)
            # iterate through all files recursively and find "genomic.fna"
            for member in the_zip.infolist():
                if member.filename.endswith("genomic.fna"):
                    handle = the_zip.open(member)
                    break
                raise ValueError("Could not find genomic.fna in zip file")
        else:
            handle = open(filename, "rt")
        for record in SeqIO.parse(handle, "fasta"):
            if mask != None:
                raise ValueError
            else:
                yield record.id, str(record.seq)[0:29891]


    def apply_mask_to_seq_iterator(self, seq_iterator, selected_indices):
        for seq_id, seq in seq_iterator:
            masked_seq = ""
            for i in selected_indices:
                masked_seq += seq[i]
            yield seq_id, masked_seq


    def apply_numpy_to_seq_iterator(self, seq_iterator):
        for seq_id, seq in seq_iterator:
            yield seq_id, self.string_to_one_hot_numpy(seq).flatten()


    def apply_mask_to_numpy_iterator(self, seq_iterator, selected_indices):
        for seq_id, flat in seq_iterator:
            masked_flat = flat[selected_indices]
            
            yield seq_id, masked_flat
    def batch_singles(self, iterator, batch_size):
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
