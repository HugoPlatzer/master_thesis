import model


def run_test(model_filename, prompt):
    m = model.GPT2Model.load_from_file(model_filename)
    answer = m.answer_prompts([prompt])[0]
    print(repr(answer))