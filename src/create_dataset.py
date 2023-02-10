'''
Pre process raw cuneiform data and create a dataset object, that will
be saved for later use.
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def labelToInteger(df):
    '''
    Takes a dataframe df as input and convert its labels into integers.
    '''
    df.loc[df.label == 'SUX','label'] = 0
    df.loc[df.label == 'OLB','label'] = 1
    df.loc[df.label == 'MPB','label'] = 2
    df.loc[df.label == 'STB','label'] = 3
    df.loc[df.label == 'NEB','label'] = 4
    df.loc[df.label == 'LTB','label'] = 5
    df.loc[df.label == 'NEA','label'] = 6


def addSpace(df):
    '''
    Takes a dataframe df as input and adds spaces between the 
    cuneiform characters in column 'text'.
    '''
    for i in range(len(df)):
        df.iloc[i,0] = " ".join( [char for char in df.iloc[i,0]] )


def createDataset():
    '''
    Performs all the pre processing procedures needed to create and 
    save the dataset.
    '''
    #Changing current working directory
    os.chdir("../data")

    #Loading data from .txt into pandas DataFrame
    train_data = pd.read_csv("raw_data/train.txt",sep='\t',
                             names=['text','label'])
    val_data   = pd.read_csv("raw_data/dev.txt", sep='\t',
                             names=['text','label'])
    test_data  = pd.read_csv("raw_data/gold.txt", sep='\t',
                             names=['text','label'])


    #In the training dataset, there are many duplicates (same text and
    #label) or entries that have the same text, but different labels.
    #This happens because, sometimes, the same piece of text may 
    #belong to different dialects. In this case, we remove all rows 
    #with duplicated texts, keeping only one copy, whose label is
    #defined to be the most frequent label among the duplicates. For 
    #example, if a certain cuneiform sentence appears 10 times
    #associated with the label SUX and 9 times with the label LTB, we 
    #eliminate 18 copies, keeping only one, whose label will be SUX
    #(the most frequent label).

    #Creates a DataFrame with all the duplicates
    text_duplicates = train_data[
        train_data.duplicated(subset='text',keep=False)]

    #In each iteration, original assumes the value of a different
    #sentence that has duplicates in the training data
    print("Preparing dataset...")
    for original in tqdm(set(text_duplicates['text'])):
        filtered_rows = text_duplicates['text'] == original
        filtered_text_duplicates = text_duplicates.loc[filtered_rows]
        most_frequent_label = filtered_text_duplicates['label'].value_counts()
        most_frequent_label = most_frequent_label.index[0]
        train_data.drop_duplicates(subset='text',keep='first',
                                   ignore_index=True, inplace=True)

        duplicates_indexes = train_data['text'] == original
        train_data.loc[duplicates_indexes, 'label'] = most_frequent_label

    train_data = train_data.reset_index(drop=True)

    #Converting labels to integers
    labelToInteger(train_data)
    labelToInteger(val_data)
    labelToInteger(test_data)

    #Adding space between cuneiform characters
    addSpace(train_data)
    addSpace(val_data)
    addSpace(test_data)

    #We concatenate the the training and validation set and the split
    #it again into new training (90 %) and validation sets (10 %)
    train_val_test = pd.concat([train_data, val_data, test_data])

    train_data08, eval_data02 = train_test_split(
        train_val_test, test_size=0.2, shuffle=True, random_state=1)
    val_data01, test_data01 = train_test_split(
        eval_data02, test_size=0.5, shuffle=True, random_state=1)

    train_data08 = train_data08.reset_index(drop=True)
    val_data01   = val_data01.reset_index(drop=True)
    test_data01  = val_data01.reset_index(drop=True)

    #Creating a Dataset object from dataframe
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(train_data08)
    dataset['val'] = Dataset.from_pandas(val_data01)
    dataset['test'] = Dataset.from_pandas(test_data01)

    #Saving dataset
    print("Saving dataset...")
    dataset.save_to_disk("datasets/cuneiform")


def main():
    print(os.getcwd())
    if not os.path.exists("../data/datasets/cuneiform"):
        if not os.path.exists("../data/datasets/"):
            os.mkdir("../data/datasets")
        createDataset()       
    else:
        print("Cuneiform dataset already exists!") 

if __name__ == '__main__':
    main()