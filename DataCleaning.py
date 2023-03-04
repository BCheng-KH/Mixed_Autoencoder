import pandas as pd
import numpy as np
import random
def setRandom(seed):
    random.seed(seed)

def clean_data(data):
    #remove unwanted columns
    data.pop("age")
    data.pop("gender")
    data.pop("accuracy")
    data.pop("country")
    data.pop("source")
    data.pop("elapsed")
    data.pop("P10")

    #replace zeros with three, questions not answered replaced with average value
    df = (data.loc[(data!=0).all(axis=1)]-3)
    for i in range(-2, 3):
        df = df.loc[~(df==i).all(axis=1)]
    return df


def split_strat(data, num_sets = 3, keys = []):
    # general idea is to split the data by question type letter(A), and then randomly select 1/3 of each letter to form
    # three mixed data sets
    # creating a dict of data frames for each letter in ASCII
    for i in range(num_sets - len(keys)):
        keys.append(f'set{str(i)}')
    columnsList = list(data)
    letterDict = {}
    for i in range (65, 81):
        letterDict[i] = []

    for i in range(65,81):
        for question in columnsList:
            if chr(i) in question:
                letterDict[i].append(question)

    for i in range (65, 81):
        random.shuffle(letterDict[i])

    # creating a dict containing three different data frames for eat data "set"
    setsDict = {}
    
    for k in keys:
        setsDict[k] = []

    # test code: works and prints the data set containing all A questions/answers
    #print (framesDict[65])

    # shuffles each dataframe so it's randomized when columns are later selected
    
    # go through each data frame, resetting a and b each time
    remainders = []
    for i in range (65, 81):
        cols = len(letterDict[i])
        increment = int((cols - cols%num_sets) /num_sets) 
        remainders += letterDict[i][(cols//num_sets) * num_sets:]
        a = 0
        b = increment 
        for s in keys:   
            # print(setsDict[s])                      # go through each of the three sets
            # print(framesDict[i].iloc[:,a:b])
            
            setsDict[s] += letterDict[i][a:b]     # and add on values from range a-b in the current letter/question dataframe
            # print(setsDict[s])  
            a += increment                                  # ^^^^ this line is probably the problem                        
            b += increment                                  # then increment a and b before moving on to next set
    cols = len(remainders)
    increment = int((cols - cols%num_sets) /num_sets) 
    a = 0
    b= increment 
    for s in keys:   
        # print(setsDict[s])                      # go through each of the three sets
        # print(framesDict[i].iloc[:,a:b])
        setsDict[s] += letterDict[i][a:b]     # and add on values from range a-b in the current letter/question dataframe
        # print(setsDict[s])  
        a += increment                                  # ^^^^ this line is probably the problem                        
        b += increment                                  # then increment a and b before moving on to next set
    return setsDict

def split_n_strat(data, num_qa = 20):
    # general idea is to split the data by question type letter(A), and then randomly select 1/3 of each letter to form
    # three mixed data sets
    # creating a dict of data frames for each letter in ASCII
    columnsList = list(data)
    letterDict = {}
    for i in range (65, 81):
        letterDict[i] = []

    for i in range(65,81):
        for question in columnsList:
            if chr(i) in question:
                letterDict[i].append(question)

    for i in range (65, 81):
        random.shuffle(letterDict[i])

    # creating a dict containing three different data frames for eat data "set"
    setQs = []
    for letter in letterDict:
        if letterDict[letter]:
            setQs.append(letterDict[letter].pop())
        if len(setQs) >= num_qa:
            break
    return setQs

   
def split(data, split_labels = None):
    if not split_labels:
        labels = list(data)
        random.shuffle(labels)
        split_labels = {"set1": labels[:50], "set2": labels[50:100], "set3": labels[100:150]}
    return {k: data[s] for k, s in split_labels.items}



def make_train_test(data, split):
    train_data = {k: data[k][:int(data[k].shape[0]*split)] for k in data}
    test_data = {k: data[k][int(data[k].shape[0]*split):] for k in data}
    return train_data, test_data

def get_input_dims(train_data):
    return {k: (np.array(train_data[k]).astype(float))[0].shape[0] for k in train_data}


def preprocessing(data, num_sets, split, keys):
    final_data = clean_data(data)
    final_data = split_strat(final_data,num_sets)
    train, test = make_train_test(final_data, split, keys)
    return train, test
