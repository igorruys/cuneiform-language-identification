from datetime import datetime
import argparse 
from datasets import load_from_disk
from transformers import (AutoTokenizer, AlbertForPreTraining, DefaultDataCollator, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling,
AlbertForMaskedLM)

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
args = parser.parse_args()

def tokenize_function(example):
    '''
    Function to be used during the tokenization of the dataset. The resulting tokenized
    dataset will have a extra column 'input_ids' with tokens ids corresponding to the 
    cuneiform charcters (and white spaces) in example.
    -------------------------------------------------------------
    Args:
        example: dictionary-like object that represents one example of the dataset,
        that is, a single pair "text","label" (Ex: {'text':..., 'label':...}).
    Returns:
        A dictionary with a single key input_ids, whose value is a list of token ids
        corresponding to the characters, given example's 'text' key.
    '''

    #output is a dictionary with keys "input_ids", "token_type_ids" and
    #"attention mask", whose values are lists of integers
    output = tokenizer(example["text"], truncation=True, max_length=max_len)
    input_batch = []
    for token_id in output['input_ids']:
        input_batch.append(token_id)
    return {"input_ids": input_batch}

#Loading dataset
dataset = load_from_disk('../data/datasets/cuneiform/')

#Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained('../tokenizers/tokenizer/')
vocab_size = tokenizer.vocab_size

#Since only 0,01% of all data has more than 64 cuneiform characters (or 2*64 -1, if we count the white spaces),
#we set max_len to this value
max_len = 2*64

#Tokenized dataset that has a single column "input_ids", that is a list of tokens ids lists.
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

#Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

config = AutoConfig.from_pretrained('albert-xlarge-v2')
model = AlbertForMaskedLM(config)

dia = datetime.today().strftime("%Y-%M-%d")
hora = datetime.now().strftime("%H-%M")

training_args = TrainingArguments(
    output_dir=f'../checkpoints/pretraining/meu_output_dia_{dia}_hora_{hora}',
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch,
    learning_rate=1e-5,
    logging_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs = args.epochs
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train']
)

trainer.train()
