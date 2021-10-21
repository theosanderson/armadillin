import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from . import modelling
import json
# Create argparser to capture input and output files
import argparse
parser = argparse.ArgumentParser(description='Create a small model for training')
parser.add_argument('-i', '--input', help='Input file', required=True)
parser.add_argument('-o', '--output', help='Output file', default=None)



args = parser.parse_args()

if not args.output:
    args.output = "./armadillin/trained_model/"
    print(f"You did not enter an output directory (use -o) so we will write to {args.output}")

if not os.path.exists(args.output):
    os.makedirs(args.output)
    

model = modelling.load_saved_model( args.input )
model, mask = modelling.create_pretrained_pruned_model(model)

output_model = f'{args.output}/model_small.h5'
output_mask = f'{args.output}/mask_small.json'
model.save(output_model)
json.dump(mask.tolist(), open(output_mask, 'wt'))

print(f"Model saved to {output_model}")
print(f"Mask saved to {output_mask}")
