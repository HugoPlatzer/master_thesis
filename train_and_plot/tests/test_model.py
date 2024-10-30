from model import GPT2Model

def test_model():
    model = GPT2Model(max_prompt_len=5, max_response_len=5, num_sequences=2,
        n_embd=128, n_layer=2, n_head=2)
    print(model)
    prompts_responses = [("1+2=", "3"), ("9+9=", "18")]
    sample = model.encode_training_sample(prompts_responses)
    print(sample)
    assert list(sample) == [0, 49, 43, 50, 61, 51,  0,  0,  0,  0,  0,
        0, 57, 43, 57, 61, 49, 56, 0,  0,  0,  0]