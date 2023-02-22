import h5py
import tqdm
import sys
import numpy as np


def prepare_sample(imgs, labels, i):
    sep_token = 260
    label_offset = 270
    img_part = imgs[i].astype(np.uint16).reshape((28*28, ))
    sep_part = np.array([sep_token], dtype=np.uint16)
    label_part = np.array([labels[i] + label_offset], dtype=np.uint16)
    sample = np.concatenate((img_part, sep_part, label_part))
    # ~ print(img_part)
    # ~ print(label_part)
    # ~ print(sample)
    return sample



in_file = sys.argv[1]
out_file = sys.argv[2]
sample_len = 28*28 + 2

in_f = h5py.File(in_file)
out_f = h5py.File(out_file, "w", driver="stdio")
imgs = np.array(in_f["img"])
labels = np.array(in_f["label"])

num_samples = imgs.shape[0]
samples = out_f.create_dataset("a", shape=(num_samples, sample_len), dtype="u2")

for i in tqdm.tqdm(range(num_samples)):
    samples[i] = prepare_sample(imgs, labels, i)
# ~ pool_size = multiprocessing.cpu_count()
# ~ pool = multiprocessing.Pool(pool_size)
# ~ samples_iter
