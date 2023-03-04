import pandas as pd
import random
base_path = "."
from pywebio.output import *
from pywebio.input import *


def pyWebInterface(Qlist):
    f = open(f'data/16PF/16.PF.questions.csv')  

    # question lookup dictionary: A1 -> 'I am comfortable around others'
    Qdict = {}
    for line in f.readlines():
        line = line.split(',',1)
        if line:
            Qdict[line[0]]=line[1]

    put_text("Please answer the following questions on a scale from 1-5, with 1 being strongly disagree, and 5 being strongly agree")
   
    new = input_group('Questions',[radio(label=(Qdict[Qlist[x]].split("rated", 1)[0]).replace('"', ''), options=[1,2,3,4,5], inline=True, name=str(x)) for x in range(0,len(Qlist))])

    put_text("Survey complete, thank you!")
    responses = pd.DataFrame()
    responses = responses.append(new, ignore_index=True)
    return responses

data = pd.read_csv(f'{base_path}/data/16PF/data.csv', sep="\t")
#Qlist = split_strat(data, num_sets = 3)
Qlist = ['A1', 'A2', 'A3']    # helpful for testing so you don't have to answer so many q's
print(pyWebInterface(Qlist))