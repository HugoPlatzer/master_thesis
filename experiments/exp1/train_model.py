from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np
import os
import pickle
from types import SimpleNamespace

data_file = "samples.dat"
data = pickle.loads(open(data_file, "rb").read())

args = TrainingArguments(
    output_dir="trainer_data/",
    overwrite_output_dir=True,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    lr_scheduler_type="linear",
    learning_rate=1e-4,
    bf16=True
)

model = GPT2LMHeadModel.from_pretrained("model/")
dummy_tokenizer = SimpleNamespace(pad_token_id=model.pad_token_id)
data_collator = DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=data
)
trainer.train()
