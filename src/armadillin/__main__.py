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

argparser.add_argument("--threshold",
                          help="Threshold for predictions",
                            type=float,
                            default=0.5)

argparser.add_argument("--custom_full_model",
                          help="Path to custom full model", type=str)

argparser.add_argument("--chunk_size",
                          help="Chunk size for predictions",
                          type=int, default=1000)

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
        
        positive = np.where(result > args.threshold)[0]
        positive_lineages = [input.all_lineages[x] for x in positive]
        levels = [input.lineage_to_level[x] for x in positive_lineages]
        max_level = max(levels)
        positive_lineages_at_max = [x for x in positive_lineages if input.lineage_to_level[x] == max_level]
        lineage_to_results = dict(zip(input.all_lineages, result))
        ordered_by_result = sorted(positive_lineages_at_max, key=lambda x: lineage_to_results[x], reverse=True)

        
        details = "" if not args.detailed_predictions else "\t" + ", ".join([
            f"{input.all_lineages[x]}:{result[x]}" for x in positive
        ])
        print(f"{sequence_names[i]}\t{ordered_by_result[0]}{details}")



if not args.custom_full_model:
    model = modelling.load_saved_model(pkg_resources.resource_filename(__name__, 'trained_model/model_small.h5.gz'))
    #model, mask = modelling.create_pretrained_pruned_model(model)
    mask = json.load(open(pkg_resources.resource_filename(__name__, 'trained_model/mask_small.json')))

else:
    model = modelling.load_saved_model(args.custom_full_model)
    mask = None



filename = args.fasta_file
input_iterator = input.yield_from_fasta(filename)
input_iterator = input.apply_numpy_to_seq_iterator(input_iterator)
if mask:
    input_iterator = input.apply_mask_to_numpy_iterator(input_iterator,
                                                    mask)


large_batch_seq_names = []
large_batch_seq_numpys = []

print("Predicting...", file=sys.stderr)
while True:
    try:
        sequence_name, sequence_numpy = next(input_iterator)
        large_batch_seq_names.append(sequence_name)
        large_batch_seq_numpys.append(sequence_numpy)
        if len(large_batch_seq_names) == args.chunk_size:
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
