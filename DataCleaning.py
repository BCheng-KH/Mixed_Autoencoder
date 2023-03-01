import pandas as pd
import random

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
    return (data.loc[(data!=0).all(axis=1)]-3)

def split_strat(data, num_sets = 3):
    # general idea is to split the data by question type letter(A), and then randomly select 1/3 of each letter to form
    # three mixed data sets
    # creating a dict of data frames for each letter in ASCII
    framesDict = {}
    for i in range(65, 81):
        framesDict[i] = pd.DataFrame()

    # running through data and sorting it into data frame based on letter
    for i in range(65, 81):
        for columns in data:
            if chr(i) in columns:
                framesDict[i][columns] = data[[columns]]

    # creating a dict containing three different data frames for eat data "set"
    setsDict = []
    #set_list = ['set1', 'set2', 'set3']
    set_list = list(range(num_sets))
    for s in set_list:
        setsDict.append(pd.DataFrame())

    # test code: works and prints the data set containing all A questions/answers
    #print (framesDict[65])

    # shuffles each dataframe so it's randomized when columns are later selected
    for i in range (65, 81):
        framesDict[i] = framesDict[i].sample(frac=1, axis=1)

    # idea here was to use iloc to select values in a range from variables a - b
    # and then increment a and b each time the "set" increments so that the data in each frame is split into the three sets
    # not working tho lol

    # calculate how many columns per "set" based on the amount of number of that letter question type
    # cols = len(framesDict[65].axes[1])
    # increment = int((cols - cols%3) /3) 

    # go through each data frame, resetting a and b each time
    remainders = pd.DataFrame()
    for i in range (65, 81):
        cols = len(framesDict[i].axes[1])
        increment = int((cols - cols%num_sets) /num_sets) 
        remainders = framesDict[i].iloc[:,(cols//num_sets) * num_sets:].join(remainders)
        a = 0
        b= increment 
        for s in set_list:   
            # print(setsDict[s])                      # go through each of the three sets
            # print(framesDict[i].iloc[:,a:b])
            
            setsDict[s] = framesDict[i].iloc[:,a:b].join(setsDict[s]  )     # and add on values from range a-b in the current letter/question dataframe
            # print(setsDict[s])  
            a += increment                                  # ^^^^ this line is probably the problem                        
            b += increment                                  # then increment a and b before moving on to next set
    cols = len(remainders.axes[1])
    increment = int((cols - cols%num_sets) /num_sets) 
    a = 0
    b= increment 
    for s in set_list:   
        # print(setsDict[s])                      # go through each of the three sets
        # print(framesDict[i].iloc[:,a:b])
        setsDict[s] = remainders.iloc[:,a:b].join(setsDict[s]  )     # and add on values from range a-b in the current letter/question dataframe
        # print(setsDict[s])  
        a += increment                                  # ^^^^ this line is probably the problem                        
        b += increment                                  # then increment a and b before moving on to next set
    return setsDict #['set1'], setsDict['set2'], setsDict['set3']]

def make_train_test(datas, split, keys):
    train_datas = {k: datas[keys[k]][:int(data.shape[0]*split)] for k in keys}
    test_datas = {k: datas[keys[k]][int(data.shape[0]*0.8):] for k in keys}
    return train_datas, test_datas

def preprocessing(data, num_sets, split, keys):
    final_data = clean_data(data)
    final_data = split_strat(final_data,num_sets)
    train, test = make_train_test(final_data, split, keys)
    return train, test
