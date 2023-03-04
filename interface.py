import pandas as pd
import random
base_path = "."
from pywebio.output import *
from pywebio.input import *

def split_strat(data, num_sets = 3):

    # creating a dict of data frames for each letter in ASCII
    framesDict = {}
    for i in range(65, 81):
        framesDict[i] = pd.DataFrame()
    
 
    # creating a dictionary of empty lists for each letter
    columnsList = data.columns.values.tolist()
    letterDict = {}
    for i in range (65, 81):
      letterDict[i] = []

    # sorting column names into list bases on letter
    for i in range(65,81):
      for question in columnsList:
        if chr(i) in question:
            letterDict[i].append(question)

    # running through data and sorting it into data frame based on letter
    for i in range(65, 81):
        for columns in data:
            if chr(i) in columns:
                framesDict[i][columns] = data[[columns]]

    # creating a dict with each key being within 0-num_sets
    setsDict = []       #value contains data frames of responses from data
    QsetsDict = {}      #value contains lists of stratified question sets
    set_list = list(range(num_sets))
    for s in set_list:
        setsDict.append(pd.DataFrame())
        QsetsDict[s] = []
   
    # shuffles each dataframe/list so it's randomized when columns are later selected
    for i in range (65, 81):
        framesDict[i] = framesDict[i].sample(frac=1, axis=1)
        random.shuffle(letterDict[i])

    #splits each letter into num_sets and adds it to dataframe or list dictionary
    remainders = pd.DataFrame()
    Qremainders = []
    for i in range (65, 81):
        cols = len(framesDict[i].axes[1])
        increment = int((cols - cols%num_sets) /num_sets) 
        remainders = framesDict[i].iloc[:,(cols//num_sets) * num_sets:].join(remainders)
        Qremainders.append(letterDict[i][(cols//num_sets) * num_sets:])
        a = 0
        b = increment 
        for s in set_list:   
            setsDict[s] = framesDict[i].iloc[:,a:b].join(setsDict[s]  )   
            QsetsDict[s].append(letterDict[i][a:b])    
            a += increment                                                       
            b += increment                                                 

    # split up the remainders from each dictionary and equally allocate amonst sets
    cols = len(remainders.axes[1])
    increment = int((cols - cols%num_sets) /num_sets) 
    a = 0
    b = increment 
    for s in set_list:  
        setsDict[s] = remainders.iloc[:,a:b].join(setsDict[s] )   
        QsetsDict[s].append(Qremainders[a:b])
        a += increment                                                     
        b += increment                                

    Qlist = sum(QsetsDict[0], [])       ## can change QsetsDict[x] to desired question set in range of num_sets
    return Qlist

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