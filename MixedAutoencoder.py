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
import json

def setRandom(seed):
    tf.random.set_seed(seed)
    random.seed(seed)

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    
    self.latent_dim = latent_dim   
    

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def set_encoder(self, encoder):
    self.encoder=encoder

  def set_decoder(self, decoder):
    self.decoder=decoder





class MixedAutoencoder():
    def __init__(self, encoders, decoders, latent_dim, input_dims, key_list, hidden_layer_shapes):
        self.encoders = encoders
        self.decoders = decoders
        self.latent_dim = latent_dim
        self.input_dims = input_dims
        self.autoencoders = []
        self.num_ae = 0
        self.pair_tuple = []
        self.key_list = key_list
        self.optimizer_encode = {e: self.optimizer() for e in self.key_list}
        self.optimizer_decode = {d: self.optimizer() for d in self.key_list}
        self.hidden_layer_shapes = hidden_layer_shapes
        
        self.mse = losses.MeanSquaredError()
    def add_set(self, encoders, decoders, input_dims, key_list, hidden_layer_shapes):
        self.encoders |= encoders
        self.decoders |= decoders
        self.input_dims |= input_dims
        self.key_list += key_list
        self.optimizer_encode |= {e: self.optimizer() for e in key_list}
        self.optimizer_decode |= {d: self.optimizer() for d in key_list}
        self.hidden_layer_shapes |= hidden_layer_shapes
    def copy(self, other):
        self.encoders = other.encoders
        self.decoders = other.decoders
        self.latent_dim = other.latent_dim
        self.input_dims = other.input_dims
        self.autoencoders = other.autoencoders
        self.num_ae = other.num_ae
        self.pair_tuple = other.pair_tuple
        self.key_list = other.key_list
        self.optimizer_encode = other.optimizer_encode
        self.optimizer_decode = other.optimizer_decode
        self.hidden_layer_shapes = other.hidden_layer_shapes
        self.mse = other.mse
    def make_pairs(self, pair_list):
        for i, j in pair_list:
            self.pair_tuple.append((i,j))
            autoencoder=Autoencoder(self.latent_dim)
            autoencoder.set_encoder(self.encoders[i])
            autoencoder.set_decoder(self.decoders[j])
            autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
            self.autoencoders.append(autoencoder)
            self.num_ae+=1
    
    def make_train_config(self, settings = {"training": [["$all", "$all", True, True]]}):
        # "training": [encoder key, decoder key, train encoder, train decoder]
        # "encoder_proximity_training": [encoder key, do training]
        # "plot": [do plot, plot dimensions, dimension axis]
        config = {
            "training": [],
            "encoder_proximity_training": [],
            "plot": [False, 2, [0, 1]]
        }
        for setting in settings:
            if setting == "plot":
                config[setting] = settings[setting]
            else:
                for st in  settings[setting]:
                    for i, s in enumerate(st):
                        if s == "$all":
                            st[i] = self.key_list
                        elif type(s) == type(""):
                            st[i] = [s]
                    if setting == "training":
                        for e in st[0]:
                            for d in st[1]:
                                config[setting].append([e, d, st[2], st[3]])
                    if setting == "encoder_proximity_training":
                        for e in st[0]:
                            config[setting].append([e, st[1]])
        return config




    def train_set(self, train, epochs, config, batch_size=32, track_num = None, validation_split = 0.2, verbose = True):
        
        train_arr = [{d: train[d].values[i] for d in self.key_list}for i in range(len(train[self.key_list[0]]))]

        random.shuffle(train_arr)
        train_data = train_arr
        #train_data = train_arr[:int(len(train_arr)*(1-validation_split))]
        #train_val = train_arr[int(len(train_arr)*(1-validation_split)):]
        
        train_batchs = [{d: np.array([train_data[n][d] for n in range(i, i+int(batch_size*(1-validation_split)))]) for d in self.key_list} for i in range((len(train_data)//batch_size)+1) if len(train_data[i:i+batch_size])]
        val_batchs = [{d: np.array([train_data[n][d] for n in range(i+int(batch_size*(1-validation_split)), i+batch_size)]) for d in self.key_list} for i in range((len(train_data)//batch_size)+1) if len(train_data[i:i+batch_size])]
        #train_val_batches = {d: [train_val[d] for t in train_val] for d in self.key_list}
        #print(train_batchs[1]["b"].shape)

        if config['plot'][0]:
            self.make_scatter(np.concatenate(self.plot_to_latent_space({k: train[k].values[track_num:track_num+1] for k in self.key_list})), config['plot'][1], config['plot'][2])
            self.make_scatter(np.concatenate(self.plot_to_latent_space({k: train[k].values for k in self.key_list})), config['plot'][1], config['plot'][2])
        for epoch in range(epochs):
            if verbose:
                print(f'epoch {epoch}')
                metrics_names = ['loss','val_loss', 'accuracy'] 
                pb_i = Progbar(len(train_batchs), stateful_metrics=metrics_names)
            order = list(range(len(config["training"])))
            random.shuffle(order)
            for i in range(len(train_batchs)):
                if epoch == i == 0:
                    tf.config.run_functions_eagerly(True)
                    loss = self.train_step(train_batchs[i], self.encoders, self.decoders, order, self.optimizer_encode, self.optimizer_decode, config)
                    tf.config.run_functions_eagerly(False)
                else:
                    loss = self.train_step(train_batchs[i], self.encoders, self.decoders, order, self.optimizer_encode, self.optimizer_decode, config)
                if verbose:
                    val, acc = self.val_step(val_batchs[i], self.encoders, self.decoders, order, config)
                    values=[('loss',loss), ('val_loss',val), ('accuracy', acc)]
                    pb_i.add(1, values=values)

            if config['plot'][0]:
                self.make_scatter(np.concatenate(self.plot_to_latent_space({k: train[k].values[track_num:track_num+1] for k in self.key_list})), config['plot'][1], config['plot'][2])
                self.make_scatter(np.concatenate(self.plot_to_latent_space({k: train[k].values for k in self.key_list})), config['plot'][1], config['plot'][2])
        plt.show()

    @tf.function
    def train_step(self, train_batch, encoders, decoders, order, optimizer_encode, optimizer_decode, config):
        egradients = {}
        dgradients = {}
        losses_list = []

        for i in order:
            e = config['training'][i][0]
            d = config['training'][i][1]
            #gradient tape tracks the gradient

            with tf.GradientTape() as etape, tf.GradientTape() as dtape:
                #print(len(tf.stack(train_batch[e])))
                

                
                pred = decoders[d](encoders[e](tf.stack(train_batch[e]), training = config['training'][i][2]), training = config['training'][i][3])
                
                losses = self.loss(tf.stack(train_batch[d]), pred)
                losses_list.append(losses)
            
            if config['training'][i][2]:
                if e in egradients.keys():
                    egradients[e].append(etape.gradient(losses,encoders[e].trainable_variables))
                else:
                    egradients[e] = [etape.gradient(losses,encoders[e].trainable_variables)]
            if config['training'][i][3]:
                if d in dgradients.keys():
                    dgradients[d].append(dtape.gradient(losses,decoders[d].trainable_variables))
                else:
                    dgradients[d] = [dtape.gradient(losses,decoders[d].trainable_variables)]
        if config['encoder_proximity_training']:
            with tf.GradientTape(persistent=True) as tape:
                enc_list = {}
                for e in config['encoder_proximity_training']:
                    enc_list[e[0]] = (encoders[e[0]](tf.stack(train_batch[e[0]]), training=e[1]))
                eloss_list = {}
                for e in config['encoder_proximity_training']:
                    eloss = 0#-hloss(enc_list[e], tf.roll(enc_list[e], shift=1, axis=0))
                    for e2 in config['encoder_proximity_training']:
                        if e[0] != e2[0]:
                            eloss += self.loss(enc_list[e[0]], enc_list[e2[0]])
                    
                    eloss_list[e[0]] = eloss

            for e in config['encoder_proximity_training']:
                if e[1]:
                    egradients[e[0]].append(tape.gradient(eloss_list[e[0]], encoders[e[0]].trainable_variables))
        for e in egradients:
            for gradient in egradients[e]:
                optimizer_encode[e].apply_gradients(zip(gradient, encoders[e].trainable_variables))
        for d in dgradients:
            for gradient in dgradients[d]:
                
                optimizer_decode[d].apply_gradients(zip(gradient, decoders[d].trainable_variables))
        return sum(losses_list)/len(losses_list)

    @tf.function
    def val_step(self, val_batch, encoders, decoders, order, config):
        losses_list = []
        acc_list = []
        for i in order:
            #gradient tape tracks the gradient
            e = config["training"][i][0]
            d = config["training"][i][1]
            pred = decoders[d](encoders[e](tf.stack(val_batch[e]), training = False), training = False)
            losses = self.loss(tf.stack(val_batch[d]), pred)
            losses_list.append(losses)
            acc_list.append(self.soft_acc(tf.cast(tf.stack(val_batch[d]), tf.float32), pred))

        return sum(losses_list)/len(losses_list), sum(acc_list)/len(acc_list)
    
    def eval_set(self,test):
        for i in range(self.num_ae):
            print(self.pair_tuple[i])
            self.autoencoders[i].evaluate(test[self.pair_tuple[i][0]].values,test[self.pair_tuple[i][1]].values)
            print(f'compared to average: {np.mean(np.square(np.array(test[self.pair_tuple[i][1]].values)))}')

    def plot_to_latent_space(self, data):
        return np.array([self.encoders[i].predict(d, verbose = False) for i, d in data.items()])

    def map_latent_space(self, res):
        mapqL = {}
        for i in self.key_list:
            points = []

            for y in range(res):
                for x in range(res):
                    points.append([x/res, y/res])
            m = self.decoders[i].predict(points, verbose = False)

            map = []
            for y in range(res):
                map.append([m[y*res+x] for x in range(res)])

            mapq = [[[x[q] for x in y] for y in map[::-1]] for q in range(self.input_dims[i])]
            mapqL[i] = np.array(mapq)
        return

    def make_scatter(self, item, dim, axis):
        if dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_xlim((-1,1))
            ax.set_ylim((-1,1))
            scatter = ax.scatter(item[:, axis[0]], item[:, axis[1]], c=list(range(len(item))))
            plt.pause(0.01)
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim((-1,1))
            ax.set_ylim((-1,1))
            ax.set_zlim((-1,1))
            scatter = ax.scatter(item[:, axis[0]], item[:, axis[1]], item[:, axis[2]], c=list(range(len(item))))
            plt.pause(0.01)

    @tf.function
    def make_prediction(self, inp, e, d):
        return self.decoders[d](self.encoders[e](tf.stack(inp), training = False), training = False)
    def make_encoding(self, inp, e):
        return self.encoders[e].predict(inp)
    def make_decoding(self, inp, d):
        return self.decoders[d].predict(inp)


    def soft_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
    def wide_acc(self, y_true, y_pred):
        return (K.mean(K.equal(K.round(y_true), K.round(y_pred))) + K.mean(K.equal(tf.math.round(y_true), tf.math.floor(y_pred))) + K.mean(K.equal(tf.math.round(y_true), tf.math.ceil(y_pred))))/2.0
    def loss(self, train, pred):
        return self.mse(train, pred)
    def optimizer(self):
        return tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def get_accuracy(self, model, train, true):

        true = tf.cast(true, tf.float32)
        pred = model(train, training = False)
            
        return self.soft_acc(true, pred)
    @tf.function
    def get_wide_accuracy(self, model, train, true):

        true = tf.cast(true, tf.float32)
        pred = model(train, training = False)
            
        return self.wide_acc(true, pred)

    def show_accuracy(self, test):
        for i in range(self.num_ae):
            print(f'calculating for {self.pair_tuple[i]}')
            acc = self.get_accuracy(self.autoencoders[i], test[self.pair_tuple[i][0]], test[self.pair_tuple[i][1]])
            w_acc = self.get_wide_accuracy(self.autoencoders[i], test[self.pair_tuple[i][0]], test[self.pair_tuple[i][1]])
            print(f'accuracy of {self.pair_tuple[i]} is: {acc}\naccuracy with error is: {w_acc}')

    def show_error_distribution(self, test):
        
        for i in range(self.num_ae):
            print(f'calculating for {self.pair_tuple[i]}')
            pred = self.autoencoders[i].predict(test[self.pair_tuple[i][0]].values, verbose = False)
            errors = []
            #print(pred.shape)
            for n in range(len(pred)):
                for q in range(len(pred[n])):
                    errors.append(pred[n][q] - test[self.pair_tuple[i][1]].values[n][q])
            fig = plt.figure()
            ax = fig.add_subplot()
            hst = ax.hist(errors, bins=50)
            plt.pause(0.01)
    def show_error_accuracy(self, test):
        errors = []
        for i in range(self.num_ae):
            #print(f'calculating for {self.pair_tuple[i]}')
            pred = self.autoencoders[i].predict(test[self.pair_tuple[i][0]].values, verbose = False)
            
            #print(pred.shape)
            for n in range(len(pred)):
                for q in range(len(pred[n])):
                    errors.append(abs(pred[n][q] - test[self.pair_tuple[i][1]].values[n][q]))
        for p in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            e_min = 0
            e_max = 5
            e_mid = (e_min + e_max)/2
            while e_max - e_min > 0.05:
                acc = len([e for e in errors if e <= e_mid])/len(errors)
                if acc < p:
                    e_min = e_mid
                else:
                    e_max = e_mid
                e_mid = (e_min + e_max)/2
            print(f'To achive {p} accuracy, an error of +-{e_max} is required.')
    def total_binary_accuracy(self, test):
        errors = []
        for i in range(self.num_ae):
            if self.pair_tuple[i][0] != self.pair_tuple[i][1]:
                #print(f'calculating for {self.pair_tuple[i]}')
                
                pred = self.autoencoders[i].predict(test[self.pair_tuple[i][0]].values, verbose = False)
                
                #print(pred.shape)
                for n in range(len(pred)):
                    for q in range(len(pred[n])):
                        if test[self.pair_tuple[i][1]].values[n][q] != 0:
                            errors.append(pred[n][q] * test[self.pair_tuple[i][1]].values[n][q])
                        # else:
                        #     errors.append(int(round(pred[n][q]) == 0)*2 - 1)
        acc = len([e for e in errors if e > 0])/len(errors)
        return acc
    def show_total_binary_accuracy(self, test):
        acc = self.total_binary_accuracy(test)
        print(f'Binary accuracy: {acc}')
    def subtotal_binary_accuracy(self, keys, test):
        errors = []
        for i in range(self.num_ae):
            if self.pair_tuple[i][0] != self.pair_tuple[i][1] and self.pair_tuple[i][0] in keys and self.pair_tuple[i][1] in keys:
                #print(f'calculating for {self.pair_tuple[i]}')
                
                pred = self.autoencoders[i].predict(test[self.pair_tuple[i][0]].values, verbose = False)
                
                #print(pred.shape)
                for n in range(len(pred)):
                    for q in range(len(pred[n])):
                        if test[self.pair_tuple[i][1]].values[n][q] != 0:
                            errors.append(pred[n][q] * test[self.pair_tuple[i][1]].values[n][q])
                        # else:
                        #     errors.append(int(round(pred[n][q]) == 0)*2 - 1)
        acc = len([e for e in errors if e > 0])/len(errors)
        return acc
    def show_subtotal_binary_accuracy(self, keys, test):
        acc = self.subtotal_binary_accuracy(keys, test)
        print(f'Binary accuracy: {acc}')
    def encoder_binary_accuracy(self, keys, test):
        errors = []
        for i in range(self.num_ae):
            if self.pair_tuple[i][0] != self.pair_tuple[i][1] and self.pair_tuple[i][0] in keys:
                #print(f'calculating for {self.pair_tuple[i]}')
                
                pred = self.autoencoders[i].predict(test[self.pair_tuple[i][0]].values, verbose = False)
                
                #print(pred.shape)
                for n in range(len(pred)):
                    for q in range(len(pred[n])):
                        if test[self.pair_tuple[i][1]].values[n][q] != 0:
                            errors.append(pred[n][q] * test[self.pair_tuple[i][1]].values[n][q])
                        # else:
                        #     errors.append(int(round(pred[n][q]) == 0)*2 - 1)
        acc = len([e for e in errors if e > 0])/len(errors)
        return acc
    def decoder_binary_accuracy(self, keys, test):
        errors = []
        for i in range(self.num_ae):
            if self.pair_tuple[i][0] != self.pair_tuple[i][1] and self.pair_tuple[i][1] in keys:
                #print(f'calculating for {self.pair_tuple[i]}')
                
                pred = self.autoencoders[i].predict(test[self.pair_tuple[i][0]].values, verbose = False)
                
                #print(pred.shape)
                for n in range(len(pred)):
                    for q in range(len(pred[n])):
                        if test[self.pair_tuple[i][1]].values[n][q] != 0:
                            errors.append(pred[n][q] * test[self.pair_tuple[i][1]].values[n][q])
                        # else:
                        #     errors.append(int(round(pred[n][q]) == 0)*2 - 1)
        acc = len([e for e in errors if e > 0])/len(errors)
        return acc
    def show_binary_accuracy(self, keys, test):
        acc_e = self.encoder_binary_accuracy(keys, test)
        acc_d = self.decoder_binary_accuracy(keys, test)
        print(f'Encoder binary accuracy: {acc_e}')
        print(f'Decoder binary accuracy: {acc_d}')











class Mixer:
    def __init__(self, base_path = '.'):
        self.base_path = base_path
    def en_de_create(self, key_list, latent_dim, input_dims, hidden_layer_shapes):
        """
        Creates the amount of encoders and decoders based on 
        the amount of different input_dims given

        :param latent_dim: latent dimension
        :param input_dims: list of input dimensions
        :return: encoders: list of encoders created
        :return: decoders: list of decoders created
        """
        encoders = {}
        decoders = {}

        for k in key_list:
            cur_encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(input_dims[k],))] + [Dense(n, activation='relu') for n in hidden_layer_shapes[k]] + [Dense(latent_dim, activation='tanh')])

            cur_decoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,))] + [Dense(n, activation='relu') for n in hidden_layer_shapes[k][::-1]] + [Dense(input_dims[k], activation=None)])

            encoders[k] = cur_encoder
            decoders[k] = cur_decoder

        return (encoders, decoders)
    def make_extra_info(self, mix):
        bare = {
            'latent_dim': mix.latent_dim,
            'input_dims': mix.input_dims,
            'key_list': mix.key_list,
            'hidden_layer_shapes': mix.hidden_layer_shapes
        }
        return bare
    def make_new(self, hidden_layer_shapes, latent_dim, input_dims):
        key_list = list(hidden_layer_shapes.keys())
        encoders, decoders = self.en_de_create(key_list, latent_dim, input_dims, hidden_layer_shapes)
        mix = MixedAutoencoder(encoders, decoders, latent_dim, input_dims, key_list, hidden_layer_shapes)
        mix.make_pairs([(i, j) for i in key_list for j in key_list])
        return mix
    
    def add_new(self, mix, hidden_layer_shapes, input_dims):
        key_list = list(hidden_layer_shapes.keys())
        latent_dim =  mix.latent_dim
        encoders, decoders = self.en_de_create(key_list, latent_dim, input_dims, hidden_layer_shapes)
        mix.add_set(encoders, decoders, input_dims, key_list, hidden_layer_shapes)
        mix.make_pairs([(i, j) for i in mix.key_list for j in mix.key_list if i in key_list or j in key_list])
        return mix

    def save_to_label(self, mix, extra, label):
        key_list = mix.key_list
        for k in key_list:
            mix.encoders[k].save(f'{self.base_path}/Models/model_{label}/encoder_{k}')
            mix.decoders[k].save(f'{self.base_path}/Models/model_{label}/decoder_{k}')
        bare = self.make_extra_info(mix)
        for b in bare:
            if b not in extra:
                extra[b] = bare[b]
        with open(f'{self.base_path}/Models/model_{label}/extra.json', 'w+') as f:
            json.dump(extra, f)

    def load_from_label(self, label):
        with open(f'{self.base_path}/Models/model_{label}/extra.json', 'r') as f:
            extra = json.load(f)
        key_list, latent_dim, input_dims, hidden_layer_shapes = (extra[k] for k in ['key_list', 'latent_dim', 'input_dims', 'hidden_layer_shapes'])

        encoders = {}
        decoders = {}
        for k in key_list:
            encoders[k] = keras.models.load_model(f'{self.base_path}/Models/model_{label}/encoder_{k}')
            decoders[k] = keras.models.load_model(f'{self.base_path}/Models/model_{label}/decoder_{k}')
        mix = MixedAutoencoder(encoders, decoders, latent_dim, input_dims, key_list, hidden_layer_shapes)
        mix.make_pairs([(i, j) for i in key_list for j in key_list])
        return mix, extra

