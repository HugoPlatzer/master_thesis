from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np
import os
import h5py
import sys
import torch
from types import SimpleNamespace


data = h5py.File("../samples_digits.hdf5")["a"]
data = np.array(data, dtype=np.int32)
data = data[:10000]


# ~ args = TrainingArguments(
    # ~ output_dir="trainer_data/",
    # ~ overwrite_output_dir=True,
    # ~ save_strategy="steps",
    # ~ save_steps=5000,
    # ~ logging_steps=10,
    # ~ num_train_epochs=50,
    # ~ per_device_train_batch_size=batch_size,
    # ~ gradient_accumulation_steps=1,
    # ~ lr_scheduler_type="linear",
    # ~ learning_rate=1e-4,
    # ~ bf16=True
# ~ )


args = TrainingArguments(
    output_dir="trainer_data/",
    overwrite_output_dir=True,
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    lr_scheduler_type="constant",
    learning_rate=1e-4,
    bf16=False
)

model = GPT2LMHeadModel.from_pretrained("model/")
dummy_tokenizer = SimpleNamespace(pad_token_id=0)
data_collator = DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=data
)
trainer.train()
