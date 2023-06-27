import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import pickle
import scipy
import math
#from nlsq_flow_full import nlsq_forward, nlsq_reverse

# input_cond_len = 15
device = torch.device("cuda:0")
# batch_size=256

def arccosh(x):
	return torch.log(x + torch.sqrt(x.pow(2)-1))

def arcsinh(x):
	return torch.log(x + torch.sqrt(x.pow(2)+1))

def nlsq_forward( z2, a, b, c, d, f):
	
	arg = d*z2 + f
	denom = 1 + arg.pow(2)
	z2 = a + b*z2 + c/denom

	logdet = cpd_sum(torch.log(b - 2*c*d*arg/denom.pow(2)),dim=(1,2))

	#z = torch.cat((z1, z2), dim=1)

	return z2, logdet

def nlsq_reverse( z2, a, b, c, d, f):

	# double needed for stability. No effect on overall speed
	a = a.double()
	b = b.double()
	c = c.double()
	d = d.double()
	f = f.double()
	z2 = z2.double()

	aa = -b*d.pow(2)
	bb = (z2-a)*d.pow(2) - 2*b*d*f
	cc = (z2-a)*2*d*f - b*(1+f.pow(2))
	dd = (z2-a)*(1+f.pow(2)) - c

	p = (3*aa*cc - bb.pow(2))/(3*aa.pow(2))
	q = (2*bb.pow(3) - 9*aa*bb*cc + 27*aa.pow(2)*dd)/(27*aa.pow(3))

	t = -2*torch.abs(q)/q*torch.sqrt(torch.abs(p)/3)
	inter_term1 = -3*torch.abs(q)/(2*p)*torch.sqrt(3/torch.abs(p))
	inter_term2 = 1/3*arccosh(torch.abs(inter_term1-1)+1)
	t = t*torch.cosh(inter_term2)

	tpos = -2*torch.sqrt(torch.abs(p)/3)
	inter_term1 = 3*q/(2*p)*torch.sqrt(3/torch.abs(p))
	inter_term2 = 1/3*arcsinh(inter_term1)
	tpos = tpos*torch.sinh(inter_term2)

	t[p > 0] = tpos[p > 0]
	y = t - bb/(3*aa)

	arg = d*y + f
	denom = 1 + arg.pow(2)

	x_new = a + b*y + c/denom

	logdet = cpd_sum(torch.log(b - 2*c*d*arg/denom.pow(2)),dim=(1,2))

	z2 = y.float()
	logdet = logdet.float()

	#z = torch.cat((z1, z2), dim=1)

	return z2, logdet		


def split_feature_fc(x):
	return x[:,0:int(x.size(1)//2)], x[:,int(x.size(1)//2):]

def cpd_sum(tensor, dim=None, keepdim=False):
	if dim is None:
		# sum up all dim
		return torch.sum(tensor)
	else:
		if isinstance(dim, int):
			dim = [dim]
		dim = sorted(dim)
		for d in dim:
			tensor = tensor.sum(dim=d, keepdim=True)
		if not keepdim:
			for i, d in enumerate(dim):
				tensor.squeeze_(d-i)
		return tensor
		
class NN_net_fc(nn.Module):
	#split
	#shift
	#scale

	def __init__(self, in_channels, out_channels, hiddden_channels):
		super().__init__()
		self.fc1 = nn.Linear(in_channels, hiddden_channels)
		self.fc2 = nn.Linear(hiddden_channels, hiddden_channels)
		self.fc3 = nn.Linear(hiddden_channels, out_channels)
	
	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

'''class BiFlowLSTM(nn.Module):
	def __init__(self, input_size, hidden_size=512, num_layers=2):
		super(BiFlowLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
		out_dim=5*(input_size)
		self.fc = nn.Linear(hidden_size*2, out_dim)  # 2 for bidirection
		#self.dp = nn.Dropout(0.25)
	
	def forward(self, x, cond_len):
		# Set initial states
		#h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
		#c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
		
		# Forward propagate LSTM
		#x = self.dp(x)
		out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)#, (h0, c0)
		
		# Decode the hidden state of the last time step
		out = self.fc( out[:,cond_len:-cond_len,:] )
		return out'''

class EDFlowLSTM(nn.Module):
	def __init__(self, input_size, hidden_size=512, num_layers=2):
		super(EDFlowLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
		out_dim=5*(input_size)
		self.fc = nn.Linear(hidden_size, out_dim)  # 2 for bidirection
		#self.dp = nn.Dropout(0.25)
	
	def forward(self, x, cond_len):
		# Set initial states
		#h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
		#c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
		
		# Forward propagate LSTM
		#x = self.dp(x)
		data_len = x.size(1) - cond_len
		x = torch.cat([x,torch.zeros(x.size(0), data_len, x.size(2)).cuda()],dim=1)
		out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)#, (h0, c0)
		
		# Decode the hidden state of the last time step
		out = self.fc( out[:,-data_len:,:] )
		return out				

# NOTE added img hid 
class ConvFlow(nn.Module):
	def __init__(self, input_size, input_cond_len, hidden_size=512, num_layers=2, use_map=False, img_hid=64):
		super(ConvFlow, self).__init__()
		assert hidden_size >= 256
		if use_map:
			complete_hidden_size = hidden_size + img_hid
		else:
			complete_hidden_size = hidden_size
		
		self.hidden_size = hidden_size
		self.img_hid = img_hid

		self.num_layers = num_layers
		self.fc_embed = nn.Linear(input_size*3, complete_hidden_size).cuda()

		self.in_cond_layers = nn.ModuleList([nn.Linear(input_cond_len*2,hidden_size), nn.ReLU(),
			nn.Linear(hidden_size,hidden_size), nn.ReLU() ])

		#nn.InstanceNorm1d(hidden_size, affine=True),

		self.conv1d_layers = nn.ModuleList([nn.Conv1d(2*complete_hidden_size,complete_hidden_size,3,padding=1),nn.ReLU(),
			nn.Conv1d(complete_hidden_size,complete_hidden_size,3,dilation=1,padding=1),nn.ReLU(),
			nn.Conv1d(complete_hidden_size,complete_hidden_size,3,dilation=1,padding=1),nn.ReLU(),
			nn.Conv1d(complete_hidden_size,complete_hidden_size,3,dilation=1,padding=1),nn.ReLU(),
			nn.Conv1d(complete_hidden_size,complete_hidden_size,3,padding=1),nn.ReLU()
			])
		out_dim=5*(input_size)
		self.fc_out = nn.Linear(complete_hidden_size, out_dim)  # 2 for bidirection
		#self.dp = nn.Dropout(0.25)
	
	def forward(self, z1, cond, cond_img=None):
		# Set initial states
		#h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
		#c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
		
		# Forward propagate LSTM
		#x = self.dp(x)
		#data_len = x.size(1) - cond_len

		if isinstance(cond, tuple):
			print('cond is tuple')
			cond_coarse, cond_init = cond
			cond_init = cond_init.view(cond_init.size(0),-1)

			if cond_coarse.size(1) < 2*z1.size(1):
					cond_coarse = torch.cat([cond_coarse, cond_coarse[:,-1,:].unsqueeze(1)], dim=1)

			cond_coarse = torch.cat([cond_coarse[:,::2,:],cond_coarse[:,1::2,:]],dim=2)
			z1 = torch.cat([z1,cond_coarse], dim=2)
		else:	
			cond_init = cond#
			
			cond_init = cond_init.view(cond_init.size(0),-1)

			cond_coarse = torch.zeros(z1.size(0),z1.size(1),z1.size(2)*2).cuda()
			z1 = torch.cat([z1,cond_coarse], dim=2)

		z1 = self.fc_embed(z1)

		for layer in self.in_cond_layers:
			cond_init = layer(cond_init)

		# NOTE: added img cond concat here
		if cond_img is not None:
			# cond_img = cond_img.view(cond_img.size(0),-1)
			cond_init = torch.cat([cond_init, cond_img], dim=1)
            

		cond_init = cond_init.unsqueeze(1).repeat(1,z1.size(1),1)
		z1 = torch.cat([z1,cond_init],dim=2)
		
		
		z1 = torch.transpose(z1,2,1)
        
		for layer in self.conv1d_layers:
			z1 = layer(z1)
		
		out = torch.transpose(z1,2,1)
		
		
		
		# Decode the hidden state of the last time step
		out = self.fc_out( out[:,:,:] )
		return out

		
class NlsqCond(nn.Module):
	def __init__(self, in_dim, hiddden_channels, layer_num, input_cond_len, use_map=False, img_hid=64):
		super().__init__()
		self.LSTM_net = ConvFlow(in_dim, input_cond_len, hiddden_channels, layer_num, use_map, img_hid).cuda()#(in_channels//2 + cond_dim, out_channels, hiddden_channels)
		self.logA = math.log(8*math.sqrt(3)/9-0.05)
		self.layer_num = layer_num
		
    # TODO what are the factors for?
	def get_params(self, nn_outp, out_len):
	
		a = nn_outp[:, :out_len, 0::5] # [B, D]
		logb = nn_outp[:,:out_len, 1::5]*0.4
		B = nn_outp[:,:out_len, 2::5]*0.3
		logd = nn_outp[:,:out_len, 3::5]*0.4
		f = nn_outp[:,:out_len, 4::5]

		b = torch.exp(logb)
		d = torch.exp(logd)
		c = torch.tanh(B)*torch.exp(self.logA + logb - logd)
		
		return a, b, c, d, f	

	def split(self,x):
		if x.size(1) % 2 != 0:
			if self.layer_num % 3 == 0:
				z1 = x[:,1:,:]
				z2 = x[:,0:1,:]
			elif self.layer_num % 3 == 1:
				z1 = x[:,[0,2],:]
				z2 = x[:,1:2,:]
			else:
				z1 = x[:,0:2,:]
				z2 = x[:,2:3,:]

		else:
			z1 = x[:,0:x.size(1)//2,:]
			z2 = x[:,x.size(1)//2:,:]
		return z1, z2

	def ccat(self,z1,z2):	
		if z1.size(1) == 1 or z2.size(1) == 1:
			if self.layer_num % 3 == 0:
				#z1 = x[:,0:1,:]
				#z2 = x[:,1:,:]
				z = torch.cat([z2,z1],dim=1)
			elif self.layer_num % 3 == 1:
				z = torch.cat([z1,z2],dim=1)
				z = z[:,[0,2,1],:]
			else:
				#z1 = x[:,1:,:]
				#z2 = x[:,0:1,:]
				z = torch.cat([z1,z2],dim=1)
		else:
			#z1 = x[:,0:x.size(1)//2,:]
			#z2 = x[:,x.size(1)//2:,:]
			z = torch.cat([z1,z2],dim=1)
		return z#1, z2


	def forward_inference(self,x,cond,logdet):
		z1,z2	= self.split(x)
		#if isinstance(cond, tuple):
		# NOTE: modified line 316
		# nn_outp			= self.LSTM_net(z1, cond)

		# check if both future and context information are given
		if not isinstance(cond, tuple):
			cond_past = cond
			cond_img = None
		else:
			if isinstance(cond[1], tuple):
				cond_past = cond[1][0]
				cond_img = cond[1][1]
				cond_fut = cond[0]
			else:
				if len(cond[1].shape) == 2:
					cond_past = cond[0]
					cond_img = cond[1]
				else:
					cond_past = cond[1]
					cond_fut = cond[0]
					cond_img = None
		
		# NOTE: added additional input
		nn_outp			= self.LSTM_net(z1, cond_past, cond_img)

		a, b, c, d, f = self.get_params(nn_outp,z2.size(1))
		
		z2, _logdet = nlsq_forward( z2, a, b, c, d, f)

		z = self.ccat(z1,z2)

		logdet = _logdet + logdet
		
		return z, logdet

	def reverse_sampling(self,x,cond,logdet):
		z1,z2	= self.split(x)
		# nn_outp			= self.LSTM_net(z1, cond)
		if not isinstance(cond, tuple):
			cond_past = cond
			cond_img = None
		else:
			if isinstance(cond[1], tuple):
				cond_past = cond[1][0]
				cond_img = cond[1][1]
				cond_fut = cond[0]
			else:
				if len(cond[1].shape) == 2:
					cond_past = cond[0]
					cond_img = cond[1]
				else:
					cond_past = cond[1]
					cond_fut = cond[0]
					cond_img = None
		
		# NOTE: added additional input
		nn_outp			= self.LSTM_net(z1, cond_past, cond_img)

		a, b, c, d, f = self.get_params(nn_outp,z2.size(1))

		z2, _logdet = nlsq_reverse( z2, a, b, c, d, f)

		z = self.ccat(z1,z2)

		logdet = _logdet + logdet

		return z, logdet

	def forward(self, input, cond, logdet = 0., reverse=False):
		if not reverse:
			x, logdet = self.forward_inference(input, cond, logdet)
		else:
			x, logdet = self.reverse_sampling(input, cond, logdet)
		return x, logdet			


class Switch(nn.Module):
	def __init__(self, layer_num):
		super().__init__()
		self.layer_num = layer_num

	def split(self,x, reverse):
		if x.size(1) % 2 != 0:
			if self.layer_num % 2 == 0:
				if not reverse:
					z1 = x[:,0:1,:]
					z2 = x[:,1:,:]
				else:
					z1 = x[:,0:2,:]
					z2 = x[:,2:,:]
			else:
				if not reverse:
					z1 = x[:,0:2,:]
					z2 = x[:,2:,:]
				else:
					z1 = x[:,0:1,:]
					z2 = x[:,1:,:]
		else:
			z1 = x[:,0:x.size(1)//2,:]
			z2 = x[:,x.size(1)//2:,:]
		return z1, z2
		
	def forward(self, input, logdet=None, reverse=False):
		if not reverse:
			z1,z2	= self.split(input, reverse)
			z = torch.cat((z2, z1), dim=1)
			return z, logdet
		else:
			z1,z2	= self.split(input, reverse)
			z = torch.cat((z2, z1), dim=1)
			return z, logdet	

class FlowStep_fc(nn.Module):
	def __init__(self,in_dim, hidden_channels, layer_num, cond_dim, input_cond_len, use_map=False, img_hid=64):
		super().__init__()
		#self.cond_dim = cond_dim
		self.nlsq_cond = NlsqCond(in_dim, hidden_channels, layer_num=layer_num, input_cond_len=input_cond_len, use_map=use_map, img_hid=img_hid)#.to(device)
		self.switch = Switch(layer_num)
	
	def forward_inference(self,x,cond,logdet=0.,reverse=False):
		if x.size(1) > 3:
			x, logdet = self.switch(x, logdet, reverse)
		x, logdet = self.nlsq_cond(x, cond, logdet, reverse)
		#x, logdet = self.mixing(x, cond, logdet, reverse)
		return x, logdet

	def reverse_sampling(self,x,cond,logdet=0.,reverse=True):
		#x, logdet = self.mixing(x, cond, logdet, reverse)
		x, logdet = self.nlsq_cond(x,cond, logdet,reverse)
		if x.size(1) > 3:
			x, logdet = self.switch(x, logdet, reverse)
		return x, logdet

	def forward(self,input,cond,logdet=0., reverse=False):
		if not reverse:
			z, logdet = self.forward_inference(input,cond,logdet,reverse)
		else:
			z, logdet = self.reverse_sampling(input,cond,logdet,reverse)

		return z, logdet


class FlowNet_fc(nn.Module):
	def __init__(self, input_dim, hidden_channels, levels, input_cond_len, cond_dim=2, use_map=False, img_hid=64):

		super().__init__()
		self.layers = nn.ModuleList()
		for l_n in range(levels):
			self.layers.append(FlowStep_fc(in_dim=input_dim, hidden_channels=hidden_channels, 
				  							layer_num=l_n, cond_dim=cond_dim, input_cond_len=input_cond_len,
											use_map=use_map, img_hid=img_hid))

		self.layers = self.layers.cuda()

	def split_even_odd(self,input):
		return torch.cat([input[:,::2,:],input[:,1::2,:]],dim=1)

	def merge_even_odd(self,z):
		z_n = torch.zeros(z.size()).cuda()

		for t in range(0,z.size(1),2):
			z_n[:,t+0,:] = z[:,t//2,:]
			if t + 1 < z.size(1):
				z_n[:,t+1,:] = z[:,t//2+(z.size(1)//2),:]

		return z_n
			

	def forward(self, input, cond, logdet=0., reverse=False, eps_std=None):
		if not reverse:
			return self.encode(input, cond, logdet)
		else:
			return self.decode(input, cond, eps_std)

	def encode(self, z, cond, logdet=0.0):
		if z.size(1) > 3:
			z = self.split_even_odd(z)
		for layer in self.layers:
			z, logdet = layer(z, cond, logdet, reverse=False)
		return z, logdet

	def decode(self, z, cond, eps_std=None):
		for layer in reversed(self.layers):
			z, logdet = layer(z, cond, logdet=0, reverse=True)
		if z.size(1) > 3:	
			z = self.merge_even_odd(z)
		return z

class CondAutoregPrior(nn.Module):
	def __init__(self,batch_size,input_dim, hidden_size=512, num_layers=2):
		super().__init__()
		#self.time_steps = time_steps
		self.input_dim = input_dim
		self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
		self.fc = nn.Linear(hidden_size, input_dim*2)

	def encode(self,x,cond):
		cond_padded = torch.cat([cond,x[:,:-1,:]],dim=1)
		out, _ = self.lstm(cond_padded)
		out = self.fc(out)
		mean = out[:,cond.size(1)-1:,:self.input_dim]
		logs = out[:,cond.size(1)-1:,self.input_dim:]
		return mean, logs


	def sample(self,cond,time_steps):
		#cond_padded = torch.cat([cond,x[:,:-1,:]],dim=1)
		curr_sample = []
		out, hidden = self.lstm(cond)
		out = self.fc(out)

		mean = out[:,-2:-1,:self.input_dim]
		logs = out[:,-2:-1,self.input_dim:]

		in_x = GaussianDiag.sample(mean,logs) #out[:,-2:-1,:]
		curr_sample.append(in_x)

		for  i in range(time_steps-1):
			out, hidden = self.lstm(in_x, hidden)
			out = self.fc(out)

			mean = out[:,0:1,:self.input_dim]
			logs = out[:,0:1,self.input_dim:]

			in_x = GaussianDiag.sample(mean,logs)
			curr_sample.append(in_x)
			
		
		curr_sample = torch.cat(curr_sample, dim=1)
		return curr_sample

	def forward(self, x, cond, time_steps=1, reverse=False ):	
		if not reverse:
			return self.encode(x,cond)
		else:
			return self.sample(cond,time_steps)
		#cond_padded = torch.cat([cond,torch.zeros(cond.size(0),time_steps,cond.size(2)).cuda()],dim=1)
		#out, _ = self.lstm(cond_padded)
		#out = self.fc(out)
		#mean = out[:,time_steps:,:self.input_dim]
		#logs = out[:,time_steps:,self.input_dim:]
		#return mean, logs

class CondPrior(nn.Module):
	def __init__(self,batch_size, input_dim, out_time_steps, hidden_size=512, num_layers=2):
		super().__init__()
		#self.time_steps = time_steps
		self.input_dim = input_dim
		self.fc_embed = nn.Linear(input_dim, hidden_size) # originally had input_size

		self.conv1d_layers = nn.ModuleList([nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),
			nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),
			nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),
			nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),
			nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU()
			])

		#self.out_time_steps = out_time_steps
		out_dim=2*input_dim#*(out_time_steps)
		self.fc_out = nn.Linear(hidden_size, out_dim)


	def sample(self,x,cond):
		#cond_padded = torch.cat([cond,x[:,:-1,:]],dim=1)
		
		x = self.fc_embed(x)
		x = torch.transpose(x,2,1)

		for layer in self.conv1d_layers:
			x = layer(x)
		
		out = torch.transpose(x,2,1)
		#out = torch.mean(out, dim=1)
		out = self.fc_out(out)
		#out = out.view(-1, self.out_time_steps, 4)
		mean = out[:,:,:self.input_dim]
		logs = out[:,:,self.input_dim:]
		return mean, logs



	def forward(self, x, cond, time_steps=1, reverse=False ):	
		return self.sample(cond,time_steps)
		#cond_padded = torch.cat([cond,torch.zeros(cond.size(0),time_steps,cond.size(2)).cuda()],dim=1)
		#out, _ = self.lstm(cond_padded)
		#out = self.fc(out)
		#mean = out[:,time_steps:,:self.input_dim]
		#logs = out[:,time_steps:,self.input_dim:]
		#return mean, logs		

def viz_haar_forward(true,pred,z_coarse,z_fine):
	plt.style.use('dark_background')
	plt.figure(figsize=(8, 8))
	axes = plt.gca()
	#axes.set_xlim([-14,14])
	#axes.set_ylim([-14,14])
	axes.axis('off')	
	plt.plot((true[:,0]).tolist(),(true[:,1]).tolist(),c='w',linewidth=6)
	plt.scatter((pred[:,0]).tolist(),(pred[:,1]).tolist(),c='r',linewidth=6)
	plt.scatter((true[0,0]).tolist(),(true[0,1]).tolist(),c='w',s=100,marker='*');
	plt.plot(true[0,0].tolist(),true[0,1].tolist(),c='r',markersize=30,marker='*');
	plt.plot((z_fine[:,0]).tolist(),(z_fine[:,1]).tolist(),linewidth=2,c='g')
	plt.plot((z_coarse[:,0]).tolist(),(z_coarse[:,1]).tolist(),linewidth=2,c='b')
	plt.savefig('./viz_haar_forward.png',bbox_inches='tight')
	#except Exception:
	plt.close();	


class NLSqCF(nn.Module):

	def __init__(self,input_dim,cond_dim,S,K,C, input_cond_len, use_map=False, img_hid=64):
		super().__init__()

		self.scales = S

		self.flows = []

		for _ in range(self.scales+1):
			self.flows.append(FlowNet_fc(input_dim= input_dim,
								hidden_channels=C,
								levels=K,
								cond_dim = cond_dim,
								input_cond_len = input_cond_len,
								use_map=use_map, 
								img_hid=img_hid
								))
		self.flows = nn.ModuleList(self.flows).cuda()
		self.cond_dim = cond_dim
		self.input_dim = input_dim
		self.relu = nn.ReLU()

		#self.autoreg_prior = CondAutoregPrior(batch_size,input_dim)

		#self.init_cond_lstm = nn.LSTM(input_dim, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)

		#self.register_parameter("mean", nn.Parameter(torch.zeros(batch_size,3*(2**(L+1)),32//(2**(L)),32//(2**(L))).float(), requires_grad = True))#//(2**(L//2))
		#self.register_parameter("logs", nn.Parameter(torch.zeros(batch_size,3*(2**(L+1)),32//(2**(L)),32//(2**(L))).float(), requires_grad = True))

	def forward(self, x=None, cond=None, time_steps=1, eps_std=None, reverse=False):	
		if not reverse:
			return self.normal_flow(x,cond)
		else:
			return self.reverse_flow(x,cond,time_steps,eps_std)

	def haar_forward(self,z):
		if z.shape[1] % 2 == 0:
			z_coarse = 0.5*(z[:,0::2,:] + z[:,1::2,:])
		else:
			z_coarse = 0.5*(z[:,2::2,:] + z[:,1::2,:])
		
		z_fine = z[:,1::2,:] - z_coarse
		return z_coarse, z_fine
	
	def haar_reverse(self,z_coarse,z_fine):
		z_odd = z_fine + z_coarse
		z_even = 2*z_coarse - z_odd
		z = torch.zeros(z_odd.size(0),z_odd.size(1)*2,z_odd.size(2)).cuda()
		
		for t in range(0,z.size(1),2):
			z[:,t+0,:] = z_even[:,t//2,:]
			z[:,t+1,:] = z_odd[:,t//2,:]	

		return z

	def eo_forward(self,z):
		#z_coarse = 0.5*( + )
		#z_fine = z[:,1::2,:] - z_coarse
		return z[:,0::2,:].contiguous(), z[:,1::2,:].contiguous()
	
	def eo_reverse(self,z_coarse,z_fine):
		z_odd = z_fine #+ z_coarse
		z_even = z_coarse#2*z_coarse - z_odd
		z = torch.zeros(z_odd.size(0),z_odd.size(1)*2,z_odd.size(2)).cuda()
		
		for t in range(0,z.size(1),2):
			z[:,t+0,:] = z_even[:,t//2,:]
			z[:,t+1,:] = z_odd[:,t//2,:]	

		return z		

	def encode_init_cond(self, init_cond):
		out, _ = self.init_cond_lstm(init_cond)
		return out[:,-1,:].unsqueeze(1)	

	def normal_flow(self, x, _init_cond):
		x_shape = list(x.size())

		z = x
		
		nll = 0

		init_cond = _init_cond#self.encode_init_cond(_init_cond)
		for s in range(self.scales + 1):
			logdet = torch.zeros_like(x[:, 0, 0])
			if s < self.scales:
				z_coarse, z_fine = self.haar_forward(z)
				#if z_coarse.size(1) % 2 != 0:
				#
				'''viz_haar_forward(_init_cond[0].detach().cpu().numpy(), 
					x[0].detach().cpu().numpy(), 
					z_coarse[0].detach().cpu().numpy(), 
					z_fine[0].detach().cpu().numpy())
				sys.exit(0)'''
				cond_coarse = z_coarse
				z_fine_out, _logdet = self.flows[s](z_fine, (cond_coarse,init_cond), logdet=logdet, reverse=False)
			else:
				z_fine = z	
				z_coarse = z.clone() 
				cond_coarse = None
				z_fine_out, _logdet = self.flows[s](z_fine, init_cond, logdet=logdet, reverse=False)

			
			mean = torch.zeros(z_fine.size()).float().to(device)
			logs = torch.zeros(z_fine.size()).float().to(device)
			logpe = GaussianDiag.logp(mean, logs, z_fine_out)
			objective = _logdet + logpe
			_nll = (-objective) / float(np.log(2.)*z_fine.size(1)*z_fine.size(2))
			nll = nll + _nll
			z = z_coarse
		
		# prior
		#mean = torch.zeros(x.size()).float().to(device)
		#logs = torch.zeros(x.size()).float().to(device)
		#mean, logs = self.autoreg_prior( z, cond)

		#logpe = GaussianDiag.logp(mean, logs, z)

		#sys.exit(0)

		#objective = _objective + logpe

		y_logits = None

		#nll = (-objective) / float(np.log(2.)*x_shape[1]*x_shape[2])
		return z, nll, objective, logpe

	def reverse_flow(self, x, init_cond, time_steps, eps_std):
		with torch.no_grad():
			#mean = torch.zeros(cond.shape[0], time_steps, self.input_dim).float().to(device)
			#logs = torch.zeros(mean.size()).float().to(device)
			#mean, logs = self.autoreg_prior( None, cond, time_steps, reverse=True)
			#z = GaussianDiag.sample(mean, logs, eps_std)
			#z = self.autoreg_prior( None, cond, time_steps, reverse=True)
			init_cond = init_cond#self.encode_init_cond(init_cond)
			for s in reversed(range(self.scales + 1)):
				if s < self.scales:
					mean = torch.zeros(z.size()).float().to(device)
					logs = torch.zeros(mean.size()).float().to(device)
					z_noise = GaussianDiag.sample(mean, logs, eps_std)
					cond_coarse = z
					z_fine = self.flows[s](z_noise, (cond_coarse,init_cond), eps_std=eps_std, reverse=True)
					z = self.haar_reverse( cond_coarse, z_fine)

				else:
					if isinstance(init_cond, tuple):
						mean = torch.zeros(init_cond[0].shape[0], time_steps//(2**self.scales), self.input_dim).float().to(device)
					else:
						mean = torch.zeros(init_cond.shape[0], time_steps//(2**self.scales), self.input_dim).float().to(device)
					logs = torch.zeros(mean.size()).float().to(device)
					z_noise = GaussianDiag.sample(mean, logs, eps_std)
					cond_coarse = None
					z_fine = self.flows[s](z_noise, init_cond, eps_std=eps_std, reverse=True)
					z = z_fine
					
		return x, z


class GaussianDiag:
	Log2PI = float(np.log(2 * np.pi))

	@staticmethod
	def likelihood(mean, logs, x):
		"""
		lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
			  k = 1 (Independent)
			  Var = logs ** 2
		"""
		return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

	@staticmethod
	def logp(mean, logs, x):
		likelihood = GaussianDiag.likelihood(mean, logs, x)
		return cpd_sum(likelihood, dim=(1,2))

	@staticmethod
	def sample(mean, logs, eps_std=None):
		eps_std = eps_std or 1
		eps = torch.normal(mean=torch.zeros_like(mean),
						   std=torch.ones_like(logs) * eps_std)
		return mean + torch.exp(logs) * eps

			


class AttnCNN(torch.nn.Module):
	def __init__(self, input_cond_len, img_hid=64):
		super().__init__()
		self.conv1 = nn.Conv2d(4,32, kernel_size=3, stride=1, padding=1) 
		self.bn1 = nn.BatchNorm2d(32);
		self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(64);
		self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(128);
		self.conv4 = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(256);
		self.conv5 = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(512);
		self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self.bn6 = nn.BatchNorm2d(512);
		
		self.pool = nn.MaxPool2d(2)
		self.drop2_out = nn.Dropout2d(p=0.2);#0.20
		self.drop1_out = nn.Dropout(p=0.2);
		
		self.fc1 = nn.Linear(512, 512)
		self.bn_fc1 = nn.BatchNorm1d(512);
		self.fc2 = nn.Linear(512, img_hid) # the 64 used to be cond_dim
		self.relu = nn.ReLU()

		self.attn_fc1 = nn.Linear(input_cond_len*2,512)
		self.attn_bn1 = nn.BatchNorm1d(512)
		self.attn_fc2 = nn.Linear(512,512)
		self.attn_bn2 = nn.BatchNorm1d(512)
		self.attn_fc3 = nn.Linear(512,16*16)
		self.sftmax = nn.Softmax(dim=1)

	def get_attn_weights(self, init_seq):
		init_seq = init_seq.view(init_seq.shape[0],-1)
		x = (self.relu(self.attn_bn1(self.attn_fc1(init_seq))))
		x = (self.relu(self.attn_bn2(self.attn_fc2(x))))
		x = self.sftmax((self.attn_fc3(x)))
		return x

	def apply_attn(self, feats, attn_weights):
		feats = feats.view(feats.shape[0],512,16*16)
		attn_weights = attn_weights.unsqueeze(1)
		feats = torch.sum((feats*attn_weights),dim=2)
		return feats

		
	def forward(self, x, init_seq):
		x[:,:3,:,:] = ((x[:,:3,:,:]/256)-0.5)*2 # TODO ???
		x = self.drop2_out(self.pool(self.relu(self.bn1(self.conv1(x)))))
		x = self.drop2_out(self.relu(self.bn2(self.conv2(x))))
		x = self.drop2_out(self.pool(self.relu(self.bn3(self.conv3(x)))))
		x = self.drop2_out(self.relu(self.bn4(self.conv4(x))))
		x = self.drop2_out(self.pool(self.relu(self.bn5(self.conv5(x)))))
		x = (self.relu(self.bn6(self.conv6(x))))

		attn_weights = self.get_attn_weights(init_seq)
		x = self.apply_attn(x, attn_weights)
		
		x = x.view(init_seq.shape[0],-1)
		
		x = self.drop1_out(self.relu(self.bn_fc1(self.fc1(x))))
		x = self.fc2(x)
		
		return(x)
		

class HBAFlow(nn.Module):
	def __init__(self,cond_dim, S,K,C, input_cond_len, use_map = False, img_hid = 64):
		super().__init__()
		#self.seq_encoder = torch.load('./net_seq_encoder.pt')
		#self.seq_encoder = self.seq_encoder.to(device)

		self.glow = NLSqCF( 2, cond_dim, S, K, C, input_cond_len, use_map, img_hid).to(device)#, coupling='NLSq'
		
		self.attnCNN = AttnCNN(input_cond_len, img_hid).to(device)

		self.use_map = use_map
		self.input_cond_len = input_cond_len
		
		
	def encode(self,data_seq,data_img, max_length): 
		#Lengths are fixed for now

		data_seq = data_seq/20 #+ torch.randn(data_seq.size()).cuda()*0.05
		
		#data_seq[:,input_cond_len:,] = data_seq[:,input_cond_len:,] - data_seq[:,input_cond_len-1:-1,]
		data_seq_cond = data_seq[:,:self.input_cond_len,:]
		data_seq_pred = data_seq[:,self.input_cond_len:,:] 
        
		if self.use_map:
			img_enc = self.attnCNN(data_img, data_seq_cond)
			data_seq_cond = (data_seq_cond, img_enc)


		#sys.exit(0)	
		x = data_seq_pred#torch.cat([data_seq_pred[:,0::2,:],data_seq_pred[:,1::2,:]], dim=1)
		z, nll, logdetj, logpe = self.glow(x, data_seq_cond, reverse=False)
		
		flow_loss = nll
		
		return flow_loss, torch.mean(logdetj), torch.mean(logpe)
			

	def decode(self,data_seq,data_img, max_length):
		with torch.no_grad():
			data_seq = data_seq/20 #+ torch.randn(data_seq.size()).cuda()*0.05

			data_seq_cond = data_seq[:,:self.input_cond_len,:]

			if self.use_map:
				img_enc = self.attnCNN(data_img, data_seq_cond)
				data_seq_cond = (data_seq_cond, img_enc)

			x, z = self.glow(None,data_seq_cond, max_length-self.input_cond_len, 1.0,reverse=True)
			
			seqs_decoded = z
			
			'''x_n = torch.zeros(x.size()).cuda()

			for t in range(0,(max_length-input_cond_len),2):
				x_n[:,t+0,:] = x[:,t//2,:]
				x_n[:,t+1,:] = x[:,t//2+((max_length-input_cond_len)//2),:]'''

			#seqs_decoded = x_n
			#seqs_decoded[:,0,:] = seqs_decoded[:,0,:] + data_seq[:,input_cond_len-1,:]
			#seqs_decoded = torch.cumsum(seqs_decoded,dim=1)
			
			# data_seq = torch.cat([data_seq[:,:self.input_cond_len,:], seqs_decoded],dim=1)

			if seqs_decoded.shape[1] < (max_length-self.input_cond_len):
				last_vel = seqs_decoded[:,-1,:] - seqs_decoded[:,-2,:]
				last_vel = last_vel.unsqueeze(1)
				last_pos = seqs_decoded[:,-1,:]
				last_pos = last_pos.unsqueeze(1)
				last_pos = last_pos + last_vel
				seqs_decoded = torch.cat([seqs_decoded, last_pos],dim=1)
				
			return seqs_decoded*20
		
	def forward(self,data_seq,data_img,length,reverse=False):
		
		if not reverse:
			return self.encode(data_seq,data_img,length)
		else:
			return self.decode(data_seq,data_img,length) 






