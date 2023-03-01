import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from pprint import pprint
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans, Birch, DBSCAN, SpectralClustering, OPTICS, AgglomerativeClustering, SpectralClustering
from keras.models import Model, Sequential
from keras.layers import Dense
from keras import layers, losses
from keras.utils import Progbar
from keras import backend as K
from MixedAutoencoder import Mixer, MixedAutoencoder, setRandom
from DataCleaning import preprocessing

# important variable
latent_dim = 3
num_sets = 4
base_path = "."
label = "demo4_3_test_1"
model_shape = [16]

seed = 12345
tf.random.set_seed(seed)
random.seed(seed)
setRandom(seed)
keys = {"a":0, "b":1, "c":2, "d":3} 

data = pd.read_csv(f'data/16PF/data.csv', sep="\t")
train, test = preprocessing(data, num_sets, 0.8, keys)

input_dims = {k: (np.array(train_datas[k]).astype(float))[0].shape[0] for k in train}

mixer = Mixer()
autoencoder_set = mixer.make_new({'a': model_shape, 'b': model_shape, 'c': model_shape, 'd': model_shape}, latent_dim, input_dims, ["a", "b", "c", "d"])

settings = {
    "training": [["$all", "$all", True, True]],#[[[k1], [k2 for k2 in keys if k1 != k2], True, True] for k1 in keys],
    #"encoder_proximity_training": [["$all", True]],
    "plot": [True, 3, [0, 1, 2]]
    }
autoencoder_set.train_set(train_datas, 25, autoencoder_set.make_train_config(settings = settings), batch_size = 64, track_num = track_num)

cold = {x: list(train_datas[x]) for x in train_datas}
mixer.save_to_label(autoencoder_set, extra = {"columns" : cold}, label = label)

autoencoder_set.eval_set(test_datas)
autoencoder_set.show_accuracy(test_datas)
autoencoder_set.show_error_accuracy(test_datas)
autoencoder_set.show_binary_accuracy(test_datas)

scatter_train = autoencoder_set.plot_to_latent_space([t.values for t in train_datas])
scatter_test = autoencoder_set.plot_to_latent_space([t.values for t in test_datas])

if latent_dim == 2:
    mapqL = autoencoder_set.map_latent_space(100)
    
autoencoder_set.make_scatter(scatter_train[0], 3, [0, 1, 2])
autoencoder_set.make_scatter(scatter_train[1], 3, [0, 1, 2])
autoencoder_set.make_scatter(scatter_train[2], 3, [0, 1, 2])
autoencoder_set.make_scatter(np.concatenate(scatter_train), 3, [0, 1, 2])

sub = 243

item = np.concatenate((scatter_train[0, sub:sub+1],scatter_train[1, sub:sub+1],scatter_train[2, sub:sub+1]))
autoencoder_set.make_scatter(item, 3, [0, 1, 2])

autoencoder_set.make_scatter(scatter_test[0], 3, [0, 1, 2])
autoencoder_set.make_scatter(scatter_test[1], 3, [0, 1, 2])
autoencoder_set.make_scatter(scatter_test[2], 3, [0, 1, 2])
autoencoder_set.make_scatter(np.concatenate(scatter_test), 3, [0, 1, 2])

sub = 432

item = np.concatenate(([s[sub:sub+1] for s in scatter_test]))
print(item)
autoencoder_set.make_scatter(item, 3, [0, 1, 2])

if latent_dim == 2:
    
    for i, dim in enumerate(input_dims):
        print(i, dim)
        for qi in range(dim):
            plt.imshow(mapqL[i][qi])
            plt.colorbar()
            plt.pause(0.01)
    plt.show()