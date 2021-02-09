import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms




class Dataset2(Dataset):
    def __init__(self, dataset_path, split_path, input_shape, sequence_length, training):
        self.training = training
        self.dataset_path = dataset_path
        self.label_index = self._extract_label_mapping(split_path) #creating a dictionary that has action name as the key and action number as value
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training) # creating a list of directories where the extracted frames are saved
        self.sequence_length = sequence_length # Defining how many frames should be taken per video for training and testing
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences]))) #Getting the label names or name of the class
        self.num_classes = len(self.label_names) # Getting the number of class
        self.transform_train = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(input_shape[-1]),
                transforms.RandomHorizontalFlip(),
                #transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        ) # This is to transform the datasets to same sizes, it's basically resizing -> converting the image to Tensor image -> then normalizing the image -> composing all the transformation in a single image
        self.transform_test = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.CenterCrop(input_shape[-1]),
                #transforms.RandomHorizontalFlip(),
                #transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )

    def _extract_label_mapping(self, split_path="data/newviptrainlist"):
        """ Extracts a mapping between activity name and softmax index """
        with open(os.path.join(split_path, "classInd.txt").replace('\\','/')) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(
        self, dataset_path, split_path="data/newviptrainlist", training=True
    ):
        """ Extracts paths to sequences given the specified train / test split """
        fn = "mtrainlist003.txt" if training else "mtestlist003.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            seq_name = line.split(".")[1]
            sequence_paths += [os.path.join(dataset_path, seq_name).replace('\\','/')]
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.replace('\\','/').split('/')[-2]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        image_path = image_path.replace('\\','/')
        return int(image_path.split('/')[-1].split('.jpg')[0].split('frame')[-1])

    def _pad_to_length(self, sequence, path):
        """ Pads the video frames to the required sequence length for small videos"""
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    
    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        if not self.dataset_path in sequence_path:
            sequence_path = self.dataset_path+sequence_path
        # Sort frame sequence based on frame number 
        image_paths = sorted(glob.glob(sequence_path+'/*.jpg'), key=lambda path: self._frame_number(path))

        

        # Pad frames of videos shorter than `self.sequence_length` to length

        image_paths = self._pad_to_length(image_paths, sequence_path)
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
            start_i = np.random.randint(0, len(image_paths) - sample_interval * self.sequence_length + 1)
            flip = False
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else len(image_paths) // self.sequence_length
            flip = False
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                img_u=Image.open(image_paths[i])
                v_path = image_paths[i].replace("\\","/").split('/')
                v_path[2] = 'v'
                v_path = '/'.join(v_path)
                img_v = Image.open(v_path)
                if self.training:
                    image_tensor_u = self.transform_train(img_u)
                    image_tensor_v = self.transform_train(img_v)

                    image_tensor = torch.cat((image_tensor_u,image_tensor_v), 0)
                else:
                    image_tensor_u = self.transform_test(img_u)
                    image_tensor_v = self.transform_test(img_v)
                    image_tensor = torch.cat((image_tensor_u,image_tensor_v), 0)
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        image_sequence = torch.stack(image_sequence)
        target = self.label_index[self._activity_from_path(sequence_path)]
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
