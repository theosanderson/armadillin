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
    print("Using CPU", file=sys.stderr)


from . import modelling
from . import input
import numpy as np
import pkg_resources

print("Welcome to Armadillin", file=sys.stderr)
print("", file=sys.stderr)
input_helper = input.Input(pkg_resources.resource_filename("armadillin", "trained_model"))


def get_result(result):
    positive = np.where(result > args.threshold)[0]
    
    positive_lineages = [input_helper.all_lineages[x] for x in positive]
    positive_values = [result[x] for x in positive]
    levels = [input_helper.lineage_to_level[x] for x in positive_lineages]
    max_level = max(levels)
    
    positive_lineages_at_max = [x for x in positive_lineages if input_helper.lineage_to_level[x] == max_level]
    
    lineage_to_results = dict(zip(positive_lineages, positive_values))
    ordered_by_result = sorted(positive_lineages_at_max, key=lambda x: lineage_to_results[x], reverse=True)
    
    
    details = "" if not args.detailed_predictions else "\t"+", ".join([f"{lineage}:{lineage_to_results[lineage]}" for lineage in positive_lineages])
    return ordered_by_result[0], details

def do_predictions(sequence_names, sequence_numpys):
    batched_numpys = input_helper.batch_singles(iter(sequence_numpys), 32)


    results = model.predict(batched_numpys)

    for i, result in enumerate(results):
        
        lineage, details = get_result(result)
        print(f"{sequence_names[i]}\t{lineage}{details}")



if not args.custom_full_model:
    model = modelling.load_saved_model(pkg_resources.resource_filename(__name__, 'trained_model/model_small.h5'))
    #model, mask = modelling.create_pretrained_pruned_model(model)
    mask = json.load(open(pkg_resources.resource_filename(__name__, 'trained_model/mask_small.json')))

else:
    model = modelling.load_saved_model(args.custom_full_model)
    mask = None



filename = args.fasta_file
input_iterator = input_helper.yield_from_fasta(filename)
input_iterator = input_helper.masked_iterator(input_iterator, mask)




print("Predicting...", file=sys.stderr)

def make_predictions():
    large_batch_seq_names = []
    large_batch_seq_numpys = []
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


make_predictions()

def main():
    # Sorry for this hack
    pass
