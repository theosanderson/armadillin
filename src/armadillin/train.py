

import argparse
import os

# Create the parser
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--starting_model',type=str)
parser.add_argument('--do_pruning', action='store_true')
parser.add_argument('--checkpoint_path',type=str)
parser.add_argument('--disable_gpu', action='store_true')
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--shard_dir', type=str)
args = parser.parse_args()

if args.disable_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow_model_optimization as tfmot
import pandas as pd
import lzma
from . import modelling
import tensorflow as tf
from . import input
from . import training_input


def main():
    # iterator = yield_batch_of_examples("train", 10)
    # test_batch = next(iterator)
    # print(test_batch[0].shape)
    # print(test_batch[1].shape)
    # raise ValueError()

    config = {
            "alphabet": input.alphabet,
            "all_lineages": input.all_lineages,
            "seq_length": 29891,
            "mode": "pruning_style"
        }
    learning_rate = 1e-3
    if not args.starting_model:

        model = modelling.build_model(config)
    else:
        model = modelling.load_saved_model(args.starting_model)

    if args.do_pruning:
        model = modelling.build_pruning_model(config,model,pruning_params={
    'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                         final_sparsity=0.95,
                                         begin_step=0 * 5,
                                         end_step = 400 * 100)
    })

    modelling.compile_model(model, learning_rate)
    print(model.summary())

    batch_size = 32

    # Callback to save model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{args.checkpoint_path}/checkpoint.h5",
        monitor='val_f1_m',
        verbose=1,
        mode='max',
        save_freq=400 * 10)

    # Check if checkpoint path exists and create it if not:
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    callbacks=[
           
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir="./pruning_logs"),
                checkpoint
            ]

    if args.use_wandb:

        import wandb
        from wandb.keras import WandbCallback

        config = dict(learning_rate=learning_rate,
        do_pruning=args.do_pruning,
        initial_model = args.starting_model,
        checkpoint_path = args.checkpoint_path,

        )
        wandb.init(project="sandslash", notes="", config=config)
        callbacks.append( WandbCallback(generator=training_input.yield_batch_of_examples(
                   "test", batch_size, args.shard_dir),
                            validation_steps=20,
                           log_weights=True),)

    gen = training_input.yield_batch_of_examples("train", batch_size, args.shard_dir)




   
    
    for i in range(100):
        model.fit_generator(
            gen,
            steps_per_epoch=400,
            epochs=100,
            validation_data=training_input.yield_batch_of_examples("test", batch_size,  args.shard_dir),
            validation_steps=50,
            callbacks=callbacks
            )
        model.save(f"{args.checkpoint_path}/model2_{i}.h5",
                include_optimizer=True,
                save_format='h5',
                overwrite=True)

if __name__ == "__main__":
    main()
