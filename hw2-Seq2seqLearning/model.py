# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:21:58 2020

@author: DCMC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    """ Applies an attention mechanism on the output features from the decoder. """
    
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.to_weight = nn.Linear(dim, 1, bias=False)
        self._init_hidden()
    
    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)
        nn.init.xavier_normal_(self.to_weight.weight)
        
    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, feat_size = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_size).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, self.dim * 2)
        
        inputs = torch.tanh(self.linear1(inputs))
        inputs = torch.tanh(self.linear2(inputs))
        inputs = torch.tanh(self.linear3(inputs))
        o = self.to_weight(inputs)
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context
    
class EncoderRNN(nn.Module):
    def __init__(self, video_size, embed_size, input_dropout_p=0.2,\
                 rnn_dropout_p=0.5, num_layers=1, bidirectional=False):
        """
        Set the hyper-parameters and build the layers.
        
            video_size (int):
            embed_size (int): 
            dropout_p (float): 
            num_layers (int):
        """
        super(EncoderRNN, self).__init__()
        self.embed_size = embed_size
        self.embed = nn.Linear(video_size, embed_size) # video feature to embedding feature
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.gru = nn.GRU(embed_size, embed_size, num_layers, batch_first=True,\
                          dropout=rnn_dropout_p, bidirectional=bidirectional)
        # self._init_hidden()
        
    def _init_hidden(self):
       nn.init.xavier_normal(self.embed.weight)
       
    def forward(self, features):
        """
        Applies a multi-layer RNN to an input sequence.
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h     
        """
        batch_size, sequence_length, video_size = features.size()
        features = self.embed(features.view(-1, video_size)) # (batch_size * seq_len, video_size)
        features = self.input_dropout(features) 
        features = features.view(batch_size, sequence_length, self.embed_size)
        
        """
        flatten params is in-place, but you can't have two consecutive in-place ops in the graph.
        TODO: This is already performed in RNNBase, no need to do twice
        """
        self.gru.flatten_parameters() 
        output, hidden = self.gru(features)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, max_seq_length, embed_size, word_size,\
                 input_dropout_p=0.1, rnn_dropout_p=0.1, num_layers=1, bidirectional=False):
        """
        Set the hyper-parameters and build the layers.
        
        
        """
        super(DecoderRNN, self).__init__()
        self.max_seq_length = max_seq_length
        self.bidirectional_encoder = bidirectional
        self.sos_id = 1
        self.eos_id = 0
        word_size = word_size * 2 if bidirectional else word_size
        self.embed = nn.Embedding(vocab_size, word_size)
        self.gru = nn.GRU(embed_size + word_size, embed_size, num_layers, batch_first=True,\
                          dropout=rnn_dropout_p, bidirectional=bidirectional)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.attention = Attention(embed_size)
        self.output = nn.Linear(embed_size, vocab_size)

        self._init_weights()
        
    def forward(self, encoder_outputs, encoder_hidden, targets=None):
        """
        Decode image feature vectors and generates captions.
        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences
        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - e.g.
        - seq_probs.size(): torch.Size([128, 18, 2184]) means that the setence has 18 words and each word 
        -                   would be selected from 2184 size of vocab.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        seq_logprobs, seq_preds = [], []
            
        batch_size, _, _ = encoder_outputs.size()
        _, self.max_seq_length = targets.size()
        
        decoder_hidden = self._init_rnn_state(encoder_hidden)
        self.gru.flatten_parameters()
        
        targets_embed = self.embed(targets)
        for i in range(self.max_seq_length - 1):
            current_words = targets_embed[:, i]
            if decoder_hidden is not None: 
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
            else: context = torch.mean(encoder_outputs, dim=1)
            
            decoder_input = torch.cat([current_words, context], dim=1)
            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
            decoder_output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
            logprobs = F.log_softmax(self.output(decoder_output.squeeze(1)), dim=1)
            seq_logprobs.append(logprobs.unsqueeze(1))
        
        seq_logprobs = torch.cat(seq_logprobs, 1)
             
        return seq_logprobs, seq_preds
    
    def no_teacher(self, encoder_outputs, encoder_hidden):
        """
        
        """
        
        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)
        
        seq_logprobs, seq_preds = [], []
        self.gru.flatten_parameters()

        current_words = Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda()
        current_words = self.embed(current_words)        
        for t in range(self.max_seq_length - 1):            
            if decoder_hidden is not None: 
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
            else: context = torch.mean(encoder_outputs, dim=1)
                        
            decoder_input = torch.cat([current_words, context], dim=1)
            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
            decoder_output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
            logprobs = F.log_softmax(self.output(decoder_output.squeeze(1)), dim=1)
            seq_logprobs.append(logprobs.unsqueeze(1))
            
            prob_prev = torch.exp(logprobs)
            pred_words = torch.multinomial(prob_prev, 1) # sample
            sampleLogprobs = logprobs.gather(1, pred_words)
            seq_preds.append(sampleLogprobs.view(-1, 1))
            
            pred_words = pred_words.view(-1).long()
            current_words = self.embed(pred_words)   
            
            # _, pred_words = torch.max(logprobs, 1) # sample from max, i.e. topk, k = 1
            # current_words = self.embed(pred_words)   
            # seq_preds.append(pred_words.unsqueeze(1))
        
        seq_logprobs = torch.cat(seq_logprobs, 1)
        seq_preds = torch.cat(seq_preds, 1)
       
        return seq_logprobs, seq_preds        
        

    # def sample_beam(self, fc_feats, att_feats, beam_size):
    #     batch_size = fc_feats.size(0)

    #     # Project the attention feats first to reduce memory and computation comsumptions.
    #     p_att_feats = self.ctx2att(att_feats.view(-1, self.att_feat_size))
    #     p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

    #     assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
    #     seq = torch.LongTensor(self.seq_length, batch_size).zero_()
    #     seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
    #     # lets process every image independently for now, for simplicity

    #     self.done_beams = [[] for _ in range(batch_size)]
    #     for k in range(batch_size):
    #         state = self.init_hidden(beam_size)
    #         tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, self.fc_feat_size)
    #         tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
    #         tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()

    #         for t in range(1):
    #             if t == 0: # input <bos>
    #                 it = fc_feats.data.new(beam_size).long().zero_()
    #                 xt = self.embed(Variable(it, requires_grad=False))

    #             output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
    #             logprobs = F.log_softmax(self.logit(output))

    #         self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, opt=opt)
    #         seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
    #         seqLogprobs[:, k] = self.done_beams[k][0]['logps']
    #     # return the samples and their log likelihoods
    #     return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    
    def inference(self, encoder_outputs, encoder_hidden):
        """ sample the best inference"""
        
        sample_from_max = 1     # topk, k = 1
        beam_size = 1           # beam search
        temperature = 1.0       # (see knowledge distillation)    
        
        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)
        
        seq_logprobs, seq_preds = [], []
        self.gru.flatten_parameters()
        
        # if beam_size > 1:
        #     return self.sample_beam(encoder_outputs, decoder_hidden, beam_size)

        for t in range(self.max_seq_length - 1):       
            if decoder_hidden is not None:
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
            else: context = torch.mean(encoder_outputs, dim=1)                
            
            if t == 0: # first word is <'sos'>
                it = Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda()
            
            elif sample_from_max: 
                sampleLogprobs, it = torch.max(logprobs, 1)
                seq_logprobs.append(sampleLogprobs.view(-1, 1))
                it = it.view(-1).long()
            
            else: # sample according to distribution
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs)
                
                else: # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs, temperature))
               
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it)
                seq_logprobs.append(sampleLogprobs.unsqueeze(1))
                it = it.view(-1).long()
            
            seq_preds.append(it.view(-1, 1))
            
            xt = self.embed(it)  
            decoder_input = torch.cat([xt, context], dim=1)
            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
            decoder_output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
            logprobs = F.log_softmax(self.output(decoder_output.squeeze(1)), dim=1)
        
        seq_logprobs = torch.cat(seq_logprobs, 1)
        seq_preds = torch.cat(seq_preds[1:], 1)
                
        return seq_logprobs, seq_preds    
        
    
    def _init_weights(self):
        """ init the weight of some layers """
        # nn.init.xavier_normal(self.embed.weight)
        nn.init.xavier_normal_(self.output.weight)
        
    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
        
class S2VTAttentionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(S2VTAttentionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, features, targets=None, mode='no_teacher'):
        """
        Args:
            features (Variable): video features of shape [batch_size, seq_len, dim_vid]
            targets (None, optional): groung truth labels
        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """

        if mode is 'teacher_forcing':
            encoder_outputs, encoder_hidden = self.encoder(features)
            seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, targets)
                         
        elif mode is 'no_teacher':
            encoder_outputs, encoder_hidden = self.encoder(features)
            seq_prob, seq_preds = self.decoder.no_teacher(encoder_outputs, encoder_hidden)           
 
        elif mode is 'inference':
            encoder_outputs, encoder_hidden = self.encoder(features)
            seq_prob, seq_preds = self.decoder.inference(encoder_outputs, encoder_hidden)
  
        return seq_prob, seq_preds     
    