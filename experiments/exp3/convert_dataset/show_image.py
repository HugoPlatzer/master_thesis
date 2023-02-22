import h5py
import sys
import os
import matplotlib.pyplot as plt

h5_file = sys.argv[1]
idx = int(sys.argv[2])

h5_f = h5py.File(h5_file)

img_data = h5_f["img"][idx]
label = h5_f["label"][idx]

title = f"file:{os.path.basename(h5_file)} idx:{idx} label:{label}"

plt.imshow(img_data, cmap="gray", vmin=0, vmax=255)
plt.title(title)
plt.show()
