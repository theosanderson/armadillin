import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from . import modelling
import json
# Create argparser to capture input and output files
import argparse
parser = argparse.ArgumentParser(description='Create a small model for training')
parser.add_argument('-i', '--input', help='Input file', required=True)

args = parser.parse_args()

model = modelling.load_saved_model( args.input )
model, mask = modelling.create_pretrained_pruned_model(model)

output_model = '/home/theo/armadillin/git-arm/src/armadillin/trained_model/model_small.h5'
output_mask = '/home/theo/armadillin/git-arm/src/armadillin/trained_model/mask_small.json'
model.save(output_model)
json.dump(mask.tolist(), open(output_mask, 'wt'))

print(f"Model saved to {output_model}")
print(f"Mask saved to {output_mask}")
