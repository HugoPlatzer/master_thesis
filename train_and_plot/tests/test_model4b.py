import model

def run_test():
    m = model.GPT2Model.load_from_file("model.bin")
    print(m.answer_prompts(["a:", "abc:"]))