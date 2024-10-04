import model

m = model.GPT2Model.load_from_file("model.bin")
print(m.answer_prompts(["a:", "abc:"]))