import argparse
import sys
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("fasta_file",
                       help="Fasta file (must be already aligned to ref)", type=str)

argparser.add_argument("--detailed_predictions",
                       help="Save detailed predictions",
                       action="store_true")
argparser.add_argument("--disable_gpu",
                       help="Don't use the GPU",
                       action="store_true")

args = argparser.parse_args()


import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if(args.disable_gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from . import modelling
from . import input
import numpy as np
import pkg_resources

print("Welcome to Armadillin", file=sys.stderr)
print("", file=sys.stderr)


def do_predictions(sequence_names, sequence_numpys):
    batched_numpys = input.batch_singles(iter(sequence_numpys), 32)

    
    results = model.predict(batched_numpys)

    for i, result in enumerate(results):
        somewhat_positive = np.where(result > 0.1)[0]
        positive = np.where(result > 0.5)[0]
        positive_lineages = [input.all_lineages[x] for x in positive]
        sorted_by_level = sorted(positive_lineages,
                                 key=lambda x: input.lineage_to_level[x],
                                 reverse=True)
        details = "" if not args.detailed_predictions else "\t" + ", ".join([
            f"{input.all_lineages[x]}:{result[x]}" for x in somewhat_positive
        ])
        print(f"{sequence_names[i]}\t{sorted_by_level[0]}{details}")




model = modelling.load_saved_model(pkg_resources.resource_filename(__name__, 'trained_model/model_small.h5'))

#model, mask = modelling.create_pretrained_pruned_model(model)
mask = json.load(open(pkg_resources.resource_filename(__name__, 'trained_model/mask_small.json')))

filename = args.fasta_file
input_iterator = input.yield_from_fasta(filename)
input_iterator = input.apply_mask_to_seq_iterator(input_iterator,
                                                  mask)
input_iterator = input.apply_numpy_to_seq_iterator(input_iterator)

large_batch_size = 5000
large_batch_seq_names = []
large_batch_seq_numpys = []

print("Predicting...", file=sys.stderr)
while True:
    try:
        sequence_name, sequence_numpy = next(input_iterator)
        large_batch_seq_names.append(sequence_name)
        large_batch_seq_numpys.append(sequence_numpy)
        if len(large_batch_seq_names) == large_batch_size:
            do_predictions(large_batch_seq_names, large_batch_seq_numpys)
            large_batch_seq_names = []
            large_batch_seq_numpys = []
    except StopIteration:
        if len(large_batch_seq_names) > 0:
            do_predictions(large_batch_seq_names, large_batch_seq_numpys)
        break


def main():
    # Sorry for this hack
    pass
