import os, glob
from PIL import Image

import numpy as np


annotattion_images = glob.glob('./data/Abyssinian_anno/*.png')
save_to = './data/Abyssinian_anno_256'
os.makedirs(save_to, exist_ok=True)

for file in annotattion_images:
    img = Image.open(file)
    print(file)
    print(img.size)
    img = np.asarray(img)
    new_img = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                new_img[i, j] = 255
            elif img[i, j] == 3:
                new_img[i, j] = 128
            else:
                new_img[i, j] = 0
    new_img = Image.fromarray(new_img)
    new_img = new_img.convert('RGB')
    save_name = os.path.basename(file)
    new_img.save(os.path.join(save_to, save_name))
