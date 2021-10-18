import numpy as np
import tensorflow_model_optimization as tfmot
import pandas as pd
import lzma
from . import modelling
import tensorflow as tf
from . import input
from . import training_input

# iterator = yield_batch_of_examples("train", 10)
# test_batch = next(iterator)
# print(test_batch[0].shape)
# print(test_batch[1].shape)
# raise ValueError()

learning_rate = 1e-3

model = modelling.build_pruning_model({
    "alphabet": input.alphabet,
    "all_lineages": input.all_lineages,
    "seq_length": 29891,
    "mode": "pruning_style"
})
modelling.compile_model(model, learning_rate)
print(model.summary())

batch_size = 32

#import wandb
#from wandb.keras import WandbCallback

config = dict(learning_rate=learning_rate)
#wandb.init(project="sandslash", notes="", config=config)

gen = training_input.yield_batch_of_examples("train", batch_size)

# Callback to save model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/theo/sandslash/checkpoints3/checkpoint.h5",
    monitor='val_f1_m',
    verbose=1,
    mode='max',
    save_freq=400 * 10)
for i in range(100):
    model.fit_generator(
        gen,
        steps_per_epoch=400,
        epochs=100,
        validation_data=training_input.yield_batch_of_examples("test", batch_size),
        validation_steps=50,
        callbacks=[
           # WandbCallback(generator=training_input.yield_batch_of_examples(
             #   "test", batch_size),
              #            validation_steps=20,
               #           log_weights=True),
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir="./pruning_logs"),
            checkpoint
        ])
    model.save(f"/home/theo/sandslash/checkpoints3/model2_{i}.h5",
               include_optimizer=True,
               save_format='h5',
               overwrite=True)
