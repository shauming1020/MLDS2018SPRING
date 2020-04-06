# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:13:58 2020

@author: DCMC
"""

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import nltk
import pickle
from utils.vocab import Vocabulary
from utils.tools import HAHA

class HAHADataset(data.Dataset):
    """HAHA Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: video feature directory.
            json: haha annotation file path.
            vocab: vocabulary wrapper.
            transform: video feature transformer.
        """
        self.root = root
        self.haha = HAHA(json)
        self.data_pair = self.haha.data_pair
        self.vocab = vocab
        self.transform = transform          

    def __getitem__(self, index):
        assert (index < self.__len__())
        """Returns one data pair (image and caption)."""
        haha = self.haha
        vocab = self.vocab
        ann_id, sentence = self.data_pair[index]

        video_feature = np.load(os.path.join(self.root, ann_id + '.npy'))
        if self.transform is not None:
            video_feature = self.transform(video_feature)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(sentence.lower())
        sentence = []
        sentence.append(vocab('<start>'))
        sentence.extend([vocab(token) for token in tokens])
        sentence.append(vocab('<end>'))
        
        video_feature = torch.Tensor(video_feature)
        target = torch.Tensor(sentence)
        
        return video_feature, target

    def __len__(self):
        return len(self.data_pair)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (video_feature, caption). 
            - video_feature: torch tensor of shape (80, 4096).
            - caption: torch tensor of shape (batch_size, variable length).
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    video_feature, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    video_feature = torch.stack(video_feature, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return video_feature, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # HAHA caption dataset
    haha = HAHADataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for HAHA dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=haha, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

if __name__ == '__main__':
    
    root = '../data/testing_data/feat/'
    json = '../data/testing_label.json'
    vocab_path = '../data/vocab.pkl'
    transform = None
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    dataset = HAHADataset(root=root, json=json, vocab=vocab)

    dataloader = get_loader(root, json, vocab, transform, batch_size=32, shuffle=False, num_workers=0)
    
    for batch_n, batch in enumerate(dataloader):
        print('batch no: {}'.format(batch_n))
        data, label, lengths = batch

        print(label[:, :12])
        print(lengths)
        
        break