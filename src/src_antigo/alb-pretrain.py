import evaluate
import tokenizers
import numpy as np
import pandas as pd
import transformers
from datetime import datetime
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AlbertForMaskedLM, AutoConfig, TrainingArguments, Trainer


dataset = load_from_disk('../CLIdata/datasets/cuneiform-spaced-indexed')

my_tokenizer = AutoTokenizer.from_pretrained('../tokenizers/bert-base-uncased_train_val_test_maxlen_512_vocab_size_1000')
vocab_size = my_tokenizer.vocab_size
max_len = my_tokenizer.model_max_length

def tokenize_function(example):
    output = my_tokenizer(example["text"], truncation=True, max_length=max_len)
    input_batch = []
    for token_id in output['input_ids']:
        input_batch.append(token_id)
    return {"input_ids": input_batch}

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=my_tokenizer, mlm=True, mlm_probability=0.15)

config = AutoConfig.from_pretrained('albert-base-v2')
model = AlbertForMaskedLM(config)

dia = datetime.today().strftime("%Y-%M-%d")
hora = datetime.now().strftime("%H-%M")

training_args = TrainingArguments(
    output_dir=f'../checkpoints-albert/standard_{dia}_hora_{hora}',
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    logging_strategy='steps',
    logging_steps=1e3,
    save_strategy='steps',
    save_steps=10_000, 
    max_steps= 100_000
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=concatenate_datasets([tokenized_dataset['train'],tokenized_dataset['val'],tokenized_dataset['test']])
)

trainer.train()