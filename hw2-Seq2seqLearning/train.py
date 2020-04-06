# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:16:47 2020

@author: DCMC
"""


import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from utils.data_loader import get_loader 
from utils.vocab import Vocabulary
from model import EncoderRNN, DecoderRNN, S2VTAttentionModel
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, seq_probs, target, lengths):
        """
        seq_probs: shape of (batch_size, seq_len, vocab_size)
        target: shape of (batch_size, seq_len)
        lengths: shape of batch_size
        """
        
        batch_size, seq_len , vocab_size = seq_probs.size()
        
        # # truncate to the same size
        # loss_fn = nn.NLLLoss(reduce=False)
        # target = target[:, :seq_len]      
        # seq_probs = to_contiguous(seq_probs).view(-1, vocab_size)
        # target = to_contiguous(target).view(-1)
        # loss = loss_fn(seq_probs, target)
        # output = torch.sum(loss) / batch_size
        # return output
        
        loss_fn = nn.NLLLoss(reduce=False)
        predict_cat = None
        groundT_cat = None

        flag = True

        for batch in range(batch_size):
            predict      = seq_probs[batch]
            ground_truth = target[batch]
            seq_len = lengths[batch] -1
            
            # padd zeros for using no teacher mode, 
            # because the model's predict may not enough to the seq lenght.
            if len(predict) < seq_len:
                leng, vocab_size = predict.size()
                z = torch.zeros((seq_len - leng, vocab_size)).cuda()
                predict = torch.cat([predict, z])
            
            predict = predict[:seq_len]
            ground_truth = ground_truth[:seq_len]

            if flag:
                predict_cat = predict
                groundT_cat = ground_truth
                flag = False

            else:
                predict_cat = torch.cat((predict_cat, predict), dim=0)
                groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

        loss = loss_fn(predict_cat, groundT_cat)
        avg_loss = torch.sum(loss) / batch_size
        return avg_loss

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
    
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = None
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = vocab.__len__()
    
    # Build data loader
    data_loader = get_loader(args.training_feat_dir, args.training_captions, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    val_data_loader = get_loader(args.validation_feat_dir, args.validation_captions, vocab,
                                 transform, args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
    
    # Build the models
    encoder = EncoderRNN(args.video_size,
                         args.embed_size,
                         args.input_dropout_p,
                         args.rnn_dropout_p,
                         args.num_layers,
                         args.bidirectional
                         ).to(device)
    decoder = DecoderRNN(vocab_size,
                         args.max_seq_length,
                         args.embed_size,
                         args.word_size,
                         ).to(device)
    model = S2VTAttentionModel(encoder, decoder)
    
    # Loss and optimizer
    criterion = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=0)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
    
    # Train the models
    total_step = len(data_loader)
    best_loss = 999
    
    for epoch in range(args.num_epochs):
        model.train()
        exp_lr_scheduler.step()         
        for i, (features, captions, lengths) in enumerate(data_loader):
            
            # Set teacher forcing and schedule sampling
            teacher_forcing_ratio  = 0.7
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
  
            # Set mini-batch dataset
            features = features.to(device)
            captions = captions.to(device)
            
            # Forward, backward and optimize
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                seq_probs, _ = model(features, captions, mode='teacher_forcing')
            else:
                # Without teacher forcing: use its own predictions as the next input
                seq_probs, _ = model(features, mode='no_teacher')
                
            loss = criterion(seq_probs, captions[:, 1:], lengths) # elimanate <SOS>
            
            optimizer.zero_grad()
            loss.backward()
            # clip_gradient(optimizer, grad_clip=0.1)
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
        
        # Evaluation
        model.eval()
        val_loss = []
        for i, (features, captions, lengths) in enumerate(val_data_loader):
            
            # Set mini-batch dataset
            features = features.to(device)
            captions = captions.to(device)
            
            # Forward, backward and optimize
            with torch.no_grad():
                seq_probs, _ = model(features, mode='no_teacher') 
            loss = criterion(seq_probs, captions[:, 1:], lengths) # elimanate <SOS>
            val_loss.append(loss.item())     

        # Print validation info   
        val_loss = np.mean(val_loss)
        print('Epoch [{}/{}], VAL_Loss: {:.4f}, Perplexity: {:5.4f}'
              .format(epoch+1, args.num_epochs, val_loss, np.exp(val_loss))) 
        
        # Save the model checkpoints
        if (epoch+1) % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(
                args.model_path, 'checkpoint-{}.ckpt'.format(epoch+1)))
            
            if best_loss > val_loss:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'best.pth'.format(epoch+1)))  
                best_loss = val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--training_feat_dir', type=str, default='data/training_data/feat/', help='directory for training feat')
    parser.add_argument('--training_captions', type=str, default='data/training_label.json', help='path for train annotation json file')
    parser.add_argument('--validation_feat_dir', type=str, default='data/testing_data/feat/', help='directory for validation feat')
    parser.add_argument('--validation_captions', type=str, default='data/testing_label.json', help='path for validation annotation json file')  
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--video_size', type=int , default=4096, help='dimension of video features')    
    parser.add_argument('--embed_size', type=int , default=1024, help='dimension of gru embedding vectors')
    parser.add_argument('--word_size', type=int , default=1024, help='dimension of word hidden states')
    parser.add_argument('--max_seq_length', type=int , default=30, help='number of max sequence length')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in gru')
    parser.add_argument('--input_dropout_p', type=float , default=0.3, help='dropout probs of input layer')    
    parser.add_argument('--rnn_dropout_p', type=float , default=0.3, help='dropout probs of gru layer')    
    parser.add_argument('--bidirectional', type=bool , default=False, help='bidirectional gru')     
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
