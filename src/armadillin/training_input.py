from . import input
import pandas as pd
import numpy as np
import lzma
import gzip

import random


def random_sample_from_list(the_list, proportion):
    return random.sample(the_list, int(proportion * len(the_list)))


num_shards = 400
all_shards = range(num_shards)
train_shards = list(range(num_shards)) #random_sample_from_list(all_shards, 0.8)
test_shards = random_sample_from_list(all_shards, 0.1)

dropout_prob = 0.1
#ref_dropout_prob = 0.05
import gzip


class TrainingInput(object):
    def __init__(self, path):
        self.path = path
        self.input_helper = input.Input(path)
        self.lineage_cache = {}

    def get_multi_hot_from_lineage(self,lineage):
        multi_hot = np.zeros(len( self.input_helper.all_lineages), dtype=np.float32)

        while True:
            if lineage in self.input_helper.lineage_to_index:
                multi_hot[self.input_helper.lineage_to_index[lineage]] = 1
            if "." in lineage:
                subparts = lineage.split(".")
                lineage = ".".join(subparts[:-1])
                continue
            elif lineage in self.input_helper.aliases and self.input_helper.aliases[lineage] != "":
                lineage = self.input_helper.aliases[lineage]
                continue
            else:
                assert lineage == "A" or lineage == "B"
                break
        return multi_hot



    def yield_examples(self,shards):
        while True:
            for shard_num in shards:
                file = gzip.open(f"{self.path}/shard_{shard_num}.tsv.gz","rt")
                for line in file:
                    epi, seq, lineage = line.strip().split("\t")

                    lineage_numpy = self.get_multi_hot_from_lineage_with_cache(
                        lineage)
                    #print(seq)
                    yield (self.input_helper.string_to_one_hot_numpy(seq,target_length = 29891), lineage_numpy)


    


    def get_multi_hot_from_lineage_with_cache(self,lineage):
        # check if lineage is a list, and raise error if so:

        if lineage in self.lineage_cache:
            return self.lineage_cache[lineage]
        lineage_numpy = self.get_multi_hot_from_lineage(lineage)
        self.lineage_cache[lineage] = lineage_numpy
        return lineage_numpy





    def add_dropout(self,generator):
        while True:
            seq, lineage = next(generator)
            random_values = np.random.random((seq.shape[0], 1))
            seq = np.where(
                random_values > dropout_prob  #+ ref_dropout_prob
                ,
                seq,
                0)

            # seq = np.where(random_values < ref_dropout_prob, ref_numpy, seq)
            yield (seq, lineage)


    def get_typed_examples(self,type):
        if type == "train":
            return self.add_dropout(self.yield_examples(train_shards))
        elif type == "test":
            return self.yield_examples(test_shards)


    def yield_batch_of_examples(self,type, batch_size):
        example_iterator = self.get_typed_examples(type)
        while True:
            batch = [next(example_iterator) for _ in range(batch_size)]
            yield (np.stack([x[0] for x in batch]), np.stack([x[1]
                                                            for x in batch]))


