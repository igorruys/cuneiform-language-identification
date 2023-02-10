'''
Create a tokenizer with the available cuneiform text.
'''

import os
from datasets import load_from_disk
from transformers import AutoTokenizer


def createTokenizer():
    dataset = load_from_disk('../data/datasets/cuneiform')

    #Loading an existing tokenizer and training a new tokenizer from the
    #old one
    tokenizer = AutoTokenizer.from_pretrained('albert-large-v2')
    
    tokenizer = tokenizer.train_new_from_iterator(
        dataset['train']['text'] + dataset['val']['text'], vocab_size = 10_000)

    #Saving the tokenizer for later use
    tokenizer.save_pretrained("../tokenizers/tokenizer")
    print("Tokenizer created successfully!")

def main():
    if not os.path.exists("../tokenizers/tokenizer"):
        if not os.path.exists("../tokenizers"):
            os.mkdir("../tokenizers")
        createTokenizer()
    else:
        print("Tokenizer already exists!")

if __name__ == "__main__":
    main()