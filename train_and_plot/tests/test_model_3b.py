import model

def test():
    m = model.GPT2Model(max_prompt_len=8, max_response_len=8, n_embd=768, n_layer=12, n_head=12)
    print(m)
    m.save_to_file("model.bin")
    m = model.GPT2Model.load_from_file("model.bin")
    print(m.answer_prompts(["a:", "abc:"]))
