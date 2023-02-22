import transformers

model_path = "../trainer_data/checkpoint-31250"
model = transformers.GPT2LMHeadModel.from_pretrained(model_path)

