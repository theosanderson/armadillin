
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
import pylign
import sys

from collections import defaultdict
try:
    from . import helpers_compiled
except ImportError:
    print("Attempting pyximport")
    import pyximport
    pyximport.install()
    from . import helpers_compiled


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


    def extract_numpy_features(self,seq,mask_positions, mask_values):
        to_output = np.zeros(len(mask_positions))
        for index_mask in range(len(mask_positions)):
            if seq[mask_positions[index_mask]] == mask_values[index_mask]:
                to_output[index_mask] = 1
        return to_output




    def masked_iterator(self,iterator, mask):
  
        mask_positions = []
        mask_values = []

        for index_full in mask:
            mask_positions.append(index_full//len(alphabet))
            mask_values.append(alphabet[index_full%len(alphabet)])

        mask_positions = np.array(mask_positions, dtype=np.int32)
        mask_values = np.array(mask_values)
        mask_values = mask_values.view(np.int32)
     
        for seq_id, seq in iterator:
            seq = seq.lower()
            to_output = helpers_compiled.extract_numpy_features(seq.encode("utf-8"), mask_positions, mask_values)
            yield seq_id, to_output




    def get_unaliased_lineage(self, lineage):
        components = lineage.split(".")
        if components[0] in self.aliases:
            return self.aliases[components[0]] + "." + ".".join(components[1:])


    def yield_from_fasta(self, filename, already_aligned=True, max_threads = None):
        print(f"Attempting to read from fasta file {filename}", sys.stderr)
        if filename.endswith(".gz"):
            handle = gzip.open(filename, "rt")
            print("Using gzip mode", sys.stderr)
        elif filename.endswith("tar.xz"):
            tar = tarfile.open(filename, "r:xz")
            members = tar.getmembers()
            # find largest member:
            largest_member = max(members, key=lambda x: x.size)
            handle = tar.extractfile(largest_member)
            handle = io.TextIOWrapper(handle, encoding='windows-1252') 
            print("Using largest file as handle", sys.stderr)
            print("Using tarxz mode", sys.stderr)
        elif filename.endswith(".xz"):
            handle = lzma.open(filename, "rt")
            print("Using xz mode", sys.stderr)
        elif filename.endswith(".zip"):
            print("Using zip mode", sys.stderr)
            print(f"Opening {filename} as zip", sys.stderr)
            the_zip = zipfile.ZipFile(filename)
            # iterate through all files recursively and find "genomic.fna"
            for member in the_zip.infolist():
                if member.filename.endswith("genomic.fna"):
                    handle = the_zip.open(member)
                    break
                raise ValueError("Could not find genomic.fna in zip file")
        else:
            print("Using text mode", sys.stderr)
            handle = open(filename, "rt")
        if already_aligned:
            for record in SeqIO.parse(handle, "fasta"):
                yield record.id, str(record.seq)
        else:
            print("Using pylign for alignment", sys.stderr)
            reference_filename = pkg_resources.resource_filename("armadillin-model", "trained_model/reference.fa")
            print(f"Using reference file {reference_filename}", sys.stderr)
            if max_threads:
                iterator = pylign.yield_aligned(input = handle, reference = reference_filename,threads= max_threads)
            else:
                iterator =  pylign.yield_aligned(input = handle, reference = reference_filename)
            for example in iterator:
                yield example


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
