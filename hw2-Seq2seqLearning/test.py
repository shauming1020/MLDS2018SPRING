# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:08:13 2020

@author: DCMC
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils.data_loader import get_loader 
from utils.vocab import Vocabulary
from model import EncoderRNN, DecoderRNN, S2VTAttentionModel
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(vocab, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + vocab.idx2word[ix]
            else:
                break
        out.append(txt)
    return out

def postprocess(raw):
    lines = []
    for line in raw:
        line = line.split()
        line = [w.replace("<unk>", "something") for w in line]
        for _ in range(3):
            line = remove_repeat(line)
            line = remove_duplicate(line)
            line = remove_dummyword(line)
        line = ' '.join(line)
        lines.append(line)
    return lines

def remove_duplicate(line):
    length = 2
    while length < len(line):
        stack = []
        skip_flag = False
        for i in range(len(line)+1-length):
            if skip_flag:
                skip_flag = False
                continue
            if (line[i], line[i+1]) not in stack:
                stack.append((line[i], line[i+1]))
            else:
                line[i], line[i+1] = None, None
                skip_flag = True
        line = [word for word in line if word is not None]
        length += 1
    return line

def remove_dummyword(line):
    stop_words = ['a','an','and','is','at','in','into','is','of','on','to','the','then', 'from', 'with']
    while line and line[-1] in stop_words:
        line.pop()
    return_line = []
    for i, word in enumerate(line):
        if word == 'something':
            if line[max(0,i-1)] == 'a' or line[max(0,i-1)] == 'an':
                return_line.pop()
        return_line.append(word)
    return return_line

def remove_repeat(seq):
    line = []
    prev_word = None
    for word in seq:
        if word != prev_word:
            line.append(word)
            prev_word = word
    return line

def main(args):    
    transform = None
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = vocab.__len__()
    
    # Load testing data id and video feature
    video_id, video_feature = [], []
    with open(args.testing_id, 'r') as f:
        for line in f:
            ID = line.strip()
            feature = np.load(os.path.join(args.testing_feat_dir, ID + '.npy'))
            feature = torch.Tensor(feature)
            video_feature.append(feature)
            video_id.append(ID)
    
    # Build data loader    
    test_data_loader = torch.utils.data.DataLoader(dataset=video_feature, 
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers)
    
    # Build the models
    encoder = EncoderRNN(args.video_size,
                         args.embed_size,
                         args.num_layers,
                         ).to(device)
    decoder = DecoderRNN(vocab_size,
                         args.max_seq_length,
                         args.embed_size,
                         args.word_size,
                         ).to(device)
    model = S2VTAttentionModel(encoder, decoder)
    
    # Load the trained model parameters
    model.load_state_dict(torch.load(args.model_path))

    # Evaluation
    model.eval()
    results = []
    for i, features in enumerate(test_data_loader):
        # Set mini-batch dataset
        features = features.to(device)
        
        # Forward, backward and optimize
        with torch.no_grad():
            seq_probs, predicts = model(features, mode='inference')
            
        sentences = decode_sequence(vocab, predicts)
        
        for sentence in sentences:
            results.append(sentence)
            
    results = postprocess(results) 
    
    f = open(args.output_testset,'w')
    for i in range(len(results)):
        f.write(video_id[i] + ',' + results[i]+'\n')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best.pth' , help='path for loading tested models')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--testing_feat_dir', type=str, default='data/testing_data/feat/', help='directory for testing feat')
    parser.add_argument('--testing_id', type=str, default='data/testing_data/id.txt', help='path for testing annotation json file')  
    parser.add_argument('--output_testset', type=str, default='output_testset.txt', help='txt for testing output captions')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--video_size', type=int , default=4096, help='dimension of video features')    
    parser.add_argument('--embed_size', type=int , default=1024, help='dimension of gru embedding vectors')
    parser.add_argument('--word_size', type=int , default=1024, help='dimension of word hidden states')
    parser.add_argument('--max_seq_length', type=int , default=20, help='number of max sequence length')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in gru')
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
