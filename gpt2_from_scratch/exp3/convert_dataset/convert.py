import sys
import h5py
import numpy as np
import tqdm

img_file = sys.argv[1]
label_file = sys.argv[2]
out_file = sys.argv[3]

img_f = open(img_file, "rb")
label_f = open(label_file, "rb")

read_int = lambda f: int.from_bytes(f.read(4), "big")

img_magic = read_int(img_f)
if img_magic != 0x00000803:
    raise Exception("invalid magic nr")
label_magic = read_int(label_f)
if label_magic != 0x00000801:
    raise Exception("invalid magic nr")

num_records_img = read_int(img_f)
num_records_label = read_int(label_f)
if num_records_img != num_records_label:
    raise Exception(f"{num_records_img} imgs != {num_records_label} lables")
num_records = num_records_img
print(f"{num_records} images")
img_rows = read_int(img_f)
img_cols = read_int(img_f)

h5_file = h5py.File(out_file, "w", driver="stdio")
dset_img = h5_file.create_dataset("img", shape=(num_records, img_rows, img_cols), dtype="u1")
dset_label = h5_file.create_dataset("label", shape=(num_records, ), dtype="u1")

for i in tqdm.tqdm(range(num_records)):
    img_data = img_f.read(img_rows * img_cols)
    img_data_np = np.array(bytearray(img_data)).reshape(img_rows, img_cols)
    dset_img[i] = img_data_np
    dset_label[i] = label_f.read(1)[0]

print(f"img dataset shape: {dset_img.shape}")
print(f"label dataset shape: {dset_label.shape}")
