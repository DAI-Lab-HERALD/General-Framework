import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

class SeqEncoder(nn.Module):
	def __init__(self, input_dim, out_dim, embed_size, num_layers=2, lstm_size=32, normalize=False, bidirectional=False, non_linearity=None):
		super().__init__()
		
		self.lstm = nn.LSTM(embed_size, lstm_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True);

		self.h0 = torch.nn.Parameter(torch.Tensor(num_layers * (2 if bidirectional else 1),1, lstm_size), requires_grad=True)
		self.h0.data.uniform_(-0.1,0.1)
		self.c0 = torch.nn.Parameter(torch.Tensor(num_layers * (2 if bidirectional else 1),1, lstm_size), requires_grad=True)
		self.c0.data.uniform_(-0.1,0.1)
		
		self.fc_embed = nn.Linear(input_dim, embed_size)

		self.fc_out1 = nn.Linear(lstm_size * (2 if bidirectional else 1), 4*lstm_size)
		self.fc_out2 = nn.Linear(4*lstm_size, out_dim)

		self.embed_size = embed_size
		self.num_layers = num_layers

		if non_linearity == None:
			self.nl = nn.Tanh()

		self.register_buffer('scale', torch.ones(1))
		self.register_buffer('shift', torch.zeros(1,out_dim))
		self.scale_init = True
		self.normalize = normalize
		self.layer_norm = nn.LayerNorm([embed_size,])
		
	def forward_GRU(self, x, lengths, hidden):
		outs = []
		x2 = self.layer_norm(self.nl(self.fc_embed(x)))
		
		emb = pack_padded_sequence(x2, lengths, batch_first=True)
		if hidden is None:
			h0 = self.h0.repeat(1,x2.shape[0],1)
			c0 = self.c0.repeat(1,x2.shape[0],1)
			outputs, hidden = self.lstm(emb, (h0,c0))
		else:
			outputs, hidden = self.lstm(emb, hidden)
			
		output_unpack = pad_packed_sequence(outputs, batch_first=True)
		#print(output_unpack[1])
		output = self.fc_out2(self.nl(self.fc_out1(output_unpack[0])))

		if self.normalize:
			if self.training:
				with torch.no_grad():
					mean_orig = torch.mean(x.view((x.shape[0],-1)),dim=0)[None,:]
					norm_orig = torch.mean(torch.norm(x.view((x.shape[0],-1))-mean_orig,dim=1),dim=0)
					mean = torch.mean(output[:,-1,:],dim=0)[None,:]
					norm = torch.mean(torch.norm(output[:,-1,:]-mean,dim=1),dim=0)
					if self.scale_init:
						self.scale[0] = norm_orig/norm
						self.shift[:] = mean
						self.scale_init = False
					else:
						alpha = 0.97
						self.shift[:] = alpha*self.shift[:] + (1-alpha)*mean
						self.scale[0] = alpha*self.scale + (1-alpha)*( norm_orig/norm)
			
			output = (output-self.shift[:,None,:]) * self.scale + self.shift[:,None,:]

		return output, hidden

	def forward(self, x, lengths, hidden = None):
		out, hidden = self.forward_GRU(x, lengths, hidden)
		return out, hidden