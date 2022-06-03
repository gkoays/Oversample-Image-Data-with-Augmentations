import pandas as pd
import numpy as np
from config import config
import torch
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
from tqdm import tqdm
import imgaug as ia
import dataloader


def oversample():
    records = pd.read_csv(config['label_file'], header=None, names=['id', 'label'])
    images_path = [config['images_path'] + str(filename) + '.npy' for filename in records['id'].tolist()]
    labels = records['label'].tolist()

    augments = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        transforms.RandomPerspective(distortion_scale=0.1),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        RandomFlip(),
        RandomAffine(10),

        # Change the order of image channels if needed
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    index_count = list(records['id'])[-1] + 1  # get last index id of data and continue from there
    pos = sum(labels)
    neg = len(labels) - pos
    diff = abs(pos - neg)
    denominator = neg if pos > neg else pos
    upsample_label = 0 if denominator == neg else 1
    loop_time = int(diff / denominator)
    if loop_time == 0: loop_time = 1
    for i in range(loop_time):
        for idx, label in tqdm(enumerate(labels)):
            if label == upsample_label:
                img_raw = np.load(images_path[idx])
                new_img = augments(img_raw)
                new_img = new_img.numpy()
                np.save(config['new_created_images_path'] + "/" + str(index_count) + ".npy", new_img)
                records = records.append({'id': str(index_count), 'label': label}, ignore_index=True)
                index_count += 1

                # to see original and augmented images, it can be looked with debug mode one by one...
                images = [img_raw[0], new_img[0]]
                ia.imshow(np.hstack(images))

    records.to_csv(config['new_label_file'], index=False)


def main():
    print("Hello World!")
    oversample()
    # data_loader = dataloader.load_data()

if __name__ == "__main__":
    main()