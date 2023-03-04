from MixedAutoencoder import Mixer, MixedAutoencoder
import MixedAutoencoder
from DataCleaning import *
import DataCleaning
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
seed = 42
MixedAutoencoder.setRandom(seed)
DataCleaning.setRandom(seed)
base_path = "."
mixer = Mixer(base_path)

num_sets = 4
latent_dim = 3
model_shape = [16]
base_path = "."
label = f'demo_{num_sets}_{latent_dim}_[{"_".join([str(s) for s in model_shape])}]'
demo_size = 20
demo_shape = [10]

base_key_list = [f'set{str(i)}' for i in range(1, num_sets+1)]
demo_key_list = ['demo']

data = pd.read_csv(f'{base_path}/data/16PF/data.csv', sep="\t")
data = clean_data(data)
data = data.sample(frac=1)
base_column_keys = split_strat(data,num_sets, base_key_list)
demo_column_keys = {demo_key_list[0]: split_n_strat(data, demo_size)}
split_data = split(data, base_column_keys)
train, test = make_train_test(split_data, 0.8)
input_dims = get_input_dims(train)

model_shapes = {k: model_shape for k in base_key_list}
autoencoder_set = mixer.make_new(model_shapes, latent_dim, input_dims)

demo_split_data = split(data, demo_column_keys)
dtrain, dtest = make_train_test(demo_split_data, 0.8)
demo_input_dims = get_input_dims(dtrain)




demo_shapes = {k: demo_shape for k in demo_key_list}
autoencoder_set = mixer.add_new(autoencoder_set, demo_shapes, demo_input_dims)

settings = {
    "training": [[base_column_keys, base_column_keys, True, True]]#[[[k1], [k2 for k2 in keys if k1 != k2], True, True] for k1 in keys],
    #"encoder_proximity_training": [["$all", True]],
    #"plot": [True, 3, [0, 1, 2]]
}
autoencoder_set.train_set(train|dtrain, 1, autoencoder_set.make_train_config(settings = settings), batch_size = 64, verbose=True)
#autoencoder_set.show_total_binary_accuracy(test)


settings = {
    "training": [[demo_key_list, "$all", True, False], ["$all", demo_key_list, False, True]]#[[[k1], [k2 for k2 in keys if k1 != k2], True, True] for k1 in keys],
    #"encoder_proximity_training": [["$all", True]],
    #"plot": [True, 3, [0, 1, 2]]
}
autoencoder_set.train_set(train | dtrain, 20, autoencoder_set.make_train_config(settings = settings), batch_size = 64, verbose=True)
autoencoder_set.show_binary_accuracy(demo_key_list, test | dtest)

