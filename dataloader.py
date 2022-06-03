import pandas as pd
from config import config
import torch.utils.data as data
import torch
import numpy as np

class Data():
    def __init__(self, train=True):
        self.records = None
        self.image_paths = []

        if train:
            self.records = pd.read_csv(config['label_file'], header=None, names=['id', 'label'])
            self.image_paths = [config['images_path'] + str(filename) + '.npy' for filename in self.records['id'].tolist()]
            self.labels = self.records['label'].tolist()

        else:
            pass
            # You can read test records in this section if you need!!!
            # Read testing/validation data
            # self.records = pd.read_csv(config['test_label_file'] , header=None, names=['id', 'label'])
            # self.image_paths = [config['test_images_path'] + str(filename) + '.npy' for filename in self.records['id'].tolist()]

        # Total positive cases
        pos = sum(self.labels)
        # Total negative cases
        neg = len(self.labels) - pos

        print('Number of -ve samples : ', neg)
        print('Number of +ve samples : ', pos)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.records)

    def __getitem__(self, index):
        img_raw = np.load(self.image_paths[index])
        label = self.labels[index]

        if label == 1:
            label = torch.FloatTensor([1])
        elif label == 0:
            label = torch.FloatTensor([0])

        return img_raw, label



def load_data():
    print('Loading Train Dataset...')
    # Load dataset
    train_data = Data()

    train_loader = data.DataLoader(
        train_data, batch_size=1, num_workers=11, shuffle=True
    )

    # print('Loading Test Dataset of {} task...'.format(task))
    # # Load test dataset if needed
    # test_data = MRData(task, train=False)
    # test_loader = data.DataLoader(
    #     test_data, batch_size=1, num_workers=11, shuffle=False
    # )

    return train_loader  #, test_loader
