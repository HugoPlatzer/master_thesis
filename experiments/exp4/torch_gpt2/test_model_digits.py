from model import GPT2
import torch
import numpy as np
import sys
import h5py
import tqdm

def prepare_model_input(img_data):
    sep_token = 260
    sep_part = np.array([sep_token], dtype=np.int32)
    
    a = img_data.astype(np.int32).reshape((28*28, ))
    a = np.concatenate((a, sep_part))
    tensor = torch.tensor(a, dtype=torch.int32)
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
    return tensor


def get_index_range(s):
    a = int(s.split("-")[0])
    b = int(s.split("-")[1])
    return (a, b + 1)



model_file = sys.argv[1]
test_file = sys.argv[2]
index_range_s = sys.argv[3]

label_offset = 270
model = GPT2(n_pos=1024, n_vocab=300, n_hidden=768, n_heads=12, n_blocks=12)
model.load_state_dict(torch.load(model_file))
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")
test_f = h5py.File(test_file)
index_range = get_index_range(index_range_s)

num_correct, num_total = 0, 0
for i in tqdm.tqdm(range(index_range[0], index_range[1])):
    img_data = test_f["img"][i]
    img_label = test_f["label"][i]
    model_input = prepare_model_input(img_data)
    # ~ print(model_input)
    # ~ print(model_input.shape)
    # ~ exit()
    model_output = model.generate(model_input, 1)
    model_output_label = model_output[0] - label_offset
    print(i, img_label, model_output_label)
    if img_label == model_output_label:
        num_correct += 1
    num_total += 1


print(f"{num_correct}/{num_total} images correctly classified")
error_rate = 1.0 - (num_correct / num_total)
print("error rate: {:.03f}%".format(100 * error_rate))
