# bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)

bbx1 = 10
bby1 = 60
bbx2 = 10
bby2 = 60

import glob
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,10]
import cv2

# Path to data
data_folder = f"/home/aistudio/work/data/"

# Read filenames in the data folder
filenames = glob.glob(f"{data_folder}*.jpg")
# Read first 10 filenames
image_paths = filenames[:4]

image_batch = []
image_batch_labels = []
n_images = 4
print(image_paths)
for i in range(4):
    image = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)
    image_batch.append(image)
image_batch_labels=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

def rand_bbox(size, lamb):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)

    bbx1 = 10
    bby1 = 600
    bbx2 = 10
    bby2 = 600

    return bbx1, bby1, bbx2, bby2
image = cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB)
# Crop a random bounding box
lamb = 0.3
size = image.shape
print('size',size)

def generate_cutmix_image(image_batch, image_batch_labels, beta):
    c=[1,0,3,2]
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    print(f'iamhere{rand_index}')
    target_a = image_batch_labels
    target_b = np.array(image_batch_labels)[c]
    print('img.shape',image_batch[0].shape)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)
    print('bbx1',bbx1)
    print('bby1',bby1)
    print('bbx2',bbx2)
    print('bby2',bby2)
    image_batch_updated = image_batch.copy()
    image_batch_updated=np.array(image_batch_updated)
    image_batch=np.array(image_batch)
    image_batch_updated[:, bbx1:bby1, bbx2:bby2, :] = image_batch[[c], bbx1:bby1, bbx2:bby2, :]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[1] * image_batch.shape[2]))

    print(f'lam is {lam}')

    label = target_a * lam + target_b * (1. - lam)

    return image_batch_updated, label

# Generate CutMix image

input_image = image_batch[0]
image_batch_updated, image_batch_labels_updated = generate_cutmix_image(image_batch, image_batch_labels, 1.0)

# Show original images
print("Original Images")
for i in range(2):
    for j in range(2):
        plt.subplot(2,2,2*i+j+1)
        plt.imshow(image_batch[2*i+j])
plt.show()


# Show CutMix images
print("CutMix Images")
for i in range(2):
    for j in range(2):
        plt.subplot(2,2,2*i+j+1)
        plt.imshow(image_batch_updated[2*i+j])
plt.show()
