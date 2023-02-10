import argparse
import itertools
from datetime import datetime
import wandb

from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoConfig, AlbertForMaskedLM,
                          TrainingArguments, DataCollatorForLanguageModeling,
                          Trainer)

wandb.init(project="cuneiform", entity="igorruys")

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
args = parser.parse_args()


def tokenize_function(example):
    '''
    This function should be given to map for dataset tokenization.
    ------------------------------------------------------------------
    Arg:
        example: Dictionary-like object containing examples of the 
        dataset (Ex: {'text':..., 'label':...}). If the option batch 
        in map is set to False, each key of the dictionary is attached
        to one single example. Otherwise, each keys is attached to
        list of examples.
    Returns:
        output: A dictionary with keys input_ids and attention_mask.
        Its values may be a single list (for map's batch option set to
        False) or a list of lists (batch set to True).
    '''
    output = tokenizer(example["text"], truncation=True, max_length=max_len)
    del output['token_type_ids']
    return output


def create_chunks(examples):
    '''
    Creates a dataset of chunks of data.
    ------------------------------------------------------------------
    Args:
        examples (dict): Dictionary-like object, whose values are 
        lists of lists.
    Returns:
        dict_chunks (dict): A dictionary with the same keys (plus a 
        copy of the key "input_ids"), but whose values are lists of 
        chunks (lists) of same size.
    '''
    chunk_size = max_len

    #For each key (inputs_ids, attention_mask, ...), we concatenate
    #all its lists
    concatenated_examples = {k: list(itertools.chain(*examples[k])) 
                            for k in examples.keys()}
    
    #We calculate the maximum number of chunks that can be formed from
    #concatenated_examples.
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    max_num_chunks = total_length // chunk_size

    #We create chunks from concatenated_examples. If total_length is
    #not a multiple of chunk_size, then the remainder will be
    #discarded.
    dict_chunks = {k:[concatenated_examples[k][chunk_size*i:(i+1)*chunk_size]
                   for i in range(max_num_chunks)]
                   for k in concatenated_examples.keys()}

    #We create a copy of input_ids that will be used as a reference
    #for masked language modeling during training.
    dict_chunks["labels"] = dict_chunks["input_ids"].copy()
    return dict_chunks

def compute_perplexity(prediction):
    logits, labels = prediction
    perplexity = np.exp(np.max(logits))

#Loading dataset
dataset = load_from_disk('../data/datasets/cuneiform/')

#Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained('../tokenizers/tokenizer/')
vocab_size = tokenizer.vocab_size

#Since only 0,01% of all data has more than 64 cuneiform characters
#(or 2*64 -1, if we count the white spaces), we set max_len to this
#value
max_len = 2*64

#Tokenized dataset that has a "input_ids" key (containg a list of
#tokens ids) and "attention_mask" key (indicating which characters 
#should be attended).
tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                remove_columns=dataset["train"].column_names)

#Dataset with examples organized in chunks of size max_len
chuked_dataset = tokenized_dataset.map(create_chunks, batched=True)

#Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                mlm_probability=0.15)

config = AutoConfig.from_pretrained('albert-xlarge-v2')
model = AlbertForMaskedLM(config)

dia = datetime.today().strftime("%Y-%M-%d")
hora = datetime.now().strftime("%H-%M")

training_args = TrainingArguments(
    output_dir=f'../checkpoints/pretraining/meu_output_dia_{dia}_hora_{hora}',
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    #num_train_epochs=args.epochs,
    max_steps=100,
    learning_rate=1e-5,
    report_to='wandb',
    evaluation_strategy='steps',
    logging_strategy='steps',
    save_strategy='steps',
    save_steps = 10_000,
    eval_steps=10,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['val']
)

trainer.train()
