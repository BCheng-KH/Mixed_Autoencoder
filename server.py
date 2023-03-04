from MixedAutoencoder import Mixer, MixedAutoencoder
import MixedAutoencoder
from DataCleaning import *
import DataCleaning
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pywebio.output import *
from pywebio.input import *
from pywebio import start_server
import io
matplotlib.use('agg')
seed = 42
MixedAutoencoder.setRandom(seed)
DataCleaning.setRandom(seed)
base_path = "."
mixer = Mixer(base_path)

num_sets = 4
latent_dim = 7
model_shape = []
encoder = "demo"
label = f'demo_{num_sets}_{latent_dim}_[{"_".join([str(s) for s in model_shape])}]'

model, extra = mixer.load_from_label(label)
model_3d, _extra = mixer.load_from_label(label+"_3d")
#print(len(extra["columns"][encoder]))
key_list = extra["key_list"]
sets = [k for k in key_list if k != encoder]

Qlist = extra["columns"][encoder]

score_labels = {
    "-2": "unlikely",
    "-1": "unlikely",
    "0": "neutral",
    "1": "likely",
    "2": "likely"
}

Qdict = {}
with open(f'data/16PF/16.PF.questions.csv') as f:
    # question lookup dictionary: A1 -> 'I am comfortable around others'
    for line in f.readlines():
        line = line.split(',',1)
        if line:
            Qdict[line[0]]=line[1].split(" rated ", 1)[0].strip("\"")


def clip(m, M, v):
    return min(max(v, m), M)


def present(Qlist, ans):
    r_string = ''
    global Qdict


    # ask the question corresponding to the letter code, and add the user input to a dict
    r_string += "predicted answers to the following questions: \n\n"
    for x in range (0, len(Qlist)):
        r_string += Qdict[Qlist[x]] + ": " + score_labels[str(clip(-2, 2, int(round(2*ans[0][x]))))] + "\n"
    return r_string
def present_features(most, least):
    global Qdict
    r_string = ''

    # ask the question corresponding to the letter code, and add the user input to a dict
    r_string += "predicted features that most allign with you: \n\n"
    for m in most:
        r_string += Qdict[m] + "\n"
    r_string += "\n\npredicted features that least allign with you: \n\n"
    for l in least:
         r_string += Qdict[l] + "\n"
    return r_string

def plot(enc_3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    ax.set_zlim((-1,1))
    scatter = ax.scatter(enc_3d[:, 0], enc_3d[:, 1], enc_3d[:, 2])
    buf = io.BytesIO()
    fig.savefig(buf)
    return buf
    



def pyWebInterface():
    global model, Qdict, encoder, Qlist

    put_text("Please answer the following questions on a scale from 1-5, with 1 being strongly disagree, and 5 being strongly agree")
    new = input_group('Questions',[radio(label=(Qdict[q]).replace('"', ''), options=[1,2,3,4,5], inline=True, name=str(q)) for q in Qlist])
    #print(len(new.items()))
    put_text("Survey complete, thank you!")
    responses = clean_survey(pd.DataFrame(new, index=[0]))

    decoder = sets[random.randrange(num_sets)]

    enc = model.make_encoding(responses, encoder)
    enc_3d = model_3d.make_encoding(responses, encoder)
    put_image(plot(enc_3d).getvalue())
    pred = model.make_decoding(enc, decoder)
    most = np.argsort((-np.array(pred[0])))[:5]
    least = np.argsort(np.array(pred[0]))[:5]
    put_text("Results\n\n\n"+present_features([extra["columns"][decoder][int(m)] for m in most], [extra["columns"][decoder][int(l)] for l in least])+"\n\n"+present(extra["columns"][decoder], pred))
    #responses = responses.append(new, ignore_index=True)
    #return responses
#Qlist = ['A1', 'A2', 'A3']    # helpful for testing so you don't have to answer so many q's

start_server(pyWebInterface, port=8080, remote_access = True)