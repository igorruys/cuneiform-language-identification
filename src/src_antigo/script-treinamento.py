import tokenizers
from datetime import datetime
import pandas as pd
from datasets import load_dataset, load_from_disk
import transformers
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, LineByLineTextDataset, BertConfig, BertForMaskedLM, DataCollatorWithPadding, Trainer, TrainingArguments, BertForSequenceClassification,
                        AutoConfig, BertModel, BertForPreTraining, DataCollatorForLanguageModeling)

def tokenize_function(example):
    output = my_tokenizer(example["text"], truncation=True, max_length=max_len)
    input_batch = []
    for token_id in output['input_ids']:
        input_batch.append(token_id)
    return {"input_ids": input_batch}

vocab_size = 1360 #sao 1360 caracteres cuneiformes em unicode no total
max_len = 128

#carregando dataset
dataset = load_from_disk('../CLIdata/datasets/cuneiform-spaced')

#====================== preparando tokenizer com caracteres cuneiformes ==================================
with open('../CLIdata/single-char/single-chars.txt',encoding='utf8') as f:
    single_chars = f.read().split('\n')
original_bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#treinamento do tokenizer com o arquivo de caracteres
my_tokenizer = original_bert_tokenizer.train_new_from_iterator(single_chars,vocab_size = vocab_size)
#---------------------------------------------------------------------------------------------------------

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=my_tokenizer, mlm=True, mlm_probability=0.15)

config = AutoConfig.from_pretrained('bert-base-uncased',vocab_size=vocab_size, max_position_embeddings=max_len,num_hidden_layers=12)
model = BertForMaskedLM(config)

print(config)


dia = datetime.today().strftime("%Y-%M-%d")
hora = datetime.now().strftime("%H-%M")


#========================= Configurando o Trainer ==================================
training_args = TrainingArguments(
    output_dir=f'../checkpoints/meu_output_dia_{dia}_hora_{hora}',
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    logging_strategy='steps',
    logging_steps=0.5e3,
    save_strategy='steps',
    save_steps=10_000, 
    max_steps= 100_000 #200_000
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator, #o default data_collator ja era o DataCollatorWithPadding, entao nao era neceessario ter criado um data_collator anteriomente
    train_dataset=tokenized_dataset['train']
)
#---------------------------------------------------------------------------------------

trainer.train()

trainer.save_model(f'./seguranca/modelo_data_{dia}_hora_{hora}')


