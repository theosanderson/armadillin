

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
parser.add_argument('--epochs', type=int, default=None)
args = parser.parse_args()

if args.disable_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow_model_optimization as tfmot
import pandas as pd
import lzma
from . import modelling
import tensorflow as tf
from . import training_input


# Create a callback to print stuff
class PrintZeros(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer(
        "prune_low_magnitude_multiply_by_weights")
        weights = layer.get_weights()[0]
        num_zeros = np.sum(weights == 0)
        num_non_zeros = np.sum(weights != 0)
        print("Number of zeros:", num_zeros)
        print("Number of non-zeros:", num_non_zeros)
        percent = num_non_zeros / (num_zeros + num_non_zeros)
        print("Proportion of non-zeros:", percent)
        if percent<0.04:
             model_filename = "/tmp/model_zeros.h5"
             print("Saving model to", model_filename)
             self.model.save(model_filename)





def main():
    # iterator = yield_batch_of_examples("train", 10)
    # test_batch = next(iterator)
    # print(test_batch[0].shape)
    # print(test_batch[1].shape)
    # raise ValueError()

    training_input_helper = training_input.TrainingInput(args.shard_dir)

    config = {
            "alphabet": training_input.input.alphabet,
            "all_lineages": training_input_helper.input_helper.all_lineages,
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
                                         final_sparsity=0.97,
                                         begin_step=0 * 5,
                                         end_step = 400 * 5)
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
        save_freq=400 * 5)

    # Check if checkpoint path exists and create it if not:
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # Define a new callback to log the pruning history



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
        callbacks.append( WandbCallback(generator=training_input_helper.yield_batch_of_examples(
                   "test", batch_size),
                            validation_steps=20,
                           log_weights=True),)
    if args.do_pruning:
        callbacks.append(PrintZeros())

    gen = training_input_helper.yield_batch_of_examples("train", batch_size)




   
    if args.epochs is None:
        args.epochs = 1000000

    model.fit_generator(
        gen,
        steps_per_epoch=400,
        epochs=args.epochs,
        validation_data=training_input_helper.yield_batch_of_examples("test", batch_size),
        validation_steps=50,
        callbacks=callbacks
        )
       

if __name__ == "__main__":
    main()
