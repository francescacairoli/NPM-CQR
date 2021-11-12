from QR import QR
import numpy as np
import os 
import pickle
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class TrainQR():

	def __init__(self, model_name, dataset, idx = None, cal_hist_size = 50, test_hist_size = 2000, opt = "Adam", n_hidden = 50, xavier_flag = False):

		self.model_name = model_name
		self.dataset = dataset
		
		self.alpha = 0.1
		if idx:
			self.idx = idx
			self.results_path = self.model_name+"/QR_results/ID_"+idx
		else:
			rnd_idx = str(np.random.randint(0,100000))
			self.idx = rnd_idx
			self.results_path = self.model_name+"/QR_results/ID_"+rnd_idx
		os.makedirs(self.results_path, exist_ok=True)

		self.cal_hist_size = cal_hist_size
		self.test_hist_size = test_hist_size

		self.valid_set_dim = 100

		self.opt = opt
		self.xavier_flag = xavier_flag
		self.n_hidden = n_hidden


	def pinball_loss(self, pred_interval, y):

		n = len(y)
		loss = 0
		alpha_low = self.alpha/2
		alpha_high = 1-self.alpha/2
		
		for i in range(n):
			low_diff_i = y[i]-pred_interval[i,0]
			high_diff_i = y[i]-pred_interval[i,1]
			if low_diff_i > 0:
				loss += alpha_low*low_diff_i
			else:
				loss += (alpha_low-1)*low_diff_i
			if high_diff_i > 0:
				loss += alpha_high*high_diff_i
			else:
				loss += (alpha_high-1)*high_diff_i

		return loss/n

	def initialize(self):

		self.dataset.load_data()

		self.qr_model = QR(input_size = int(self.dataset.x_dim), hidden_size = self.n_hidden, xavier_flag = self.xavier_flag)

		if cuda:
			self.qr_model.cuda()


	def train(self, n_epochs, batch_size, lr):
		
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.lr = lr

		if self.opt == "Adam":
			optimizer = torch.optim.Adam(self.qr_model.parameters(), lr=lr)
		else:
			optimizer = torch.optim.RMSProp(self.qr_model.parameters(), lr=lr)

		self.net_path = self.results_path+"/qr_{}epochs.pt".format(n_epochs)

		losses = []
		val_losses = []

		bat_per_epo = self.dataset.n_training_points // batch_size
		
		Xt_val = Variable(FloatTensor(np.repeat(self.dataset.X_cal, self.cal_hist_size, axis = 0)[:(self.valid_set_dim*self.cal_hist_size)]))
		Tt_val = Variable(FloatTensor(self.dataset.R_cal[:(self.valid_set_dim*self.cal_hist_size)]))
				
		for epoch in range(n_epochs):
						
			if (epoch+1) % 25 == 0:
				print("Epoch= {},\t loss = {:2.4f}".format(epoch+1, losses[-1]))

			tmp_val_loss = []
			tmp_loss = []

			for i in range(bat_per_epo):
				# Select a minibatch
				state, rob, sign, b_ix = self.dataset.generate_mini_batches(batch_size)
				Xt = Variable(FloatTensor(state))
				Tt = Variable(FloatTensor(rob))
				
				# initialization of the gradients
				optimizer.zero_grad()
				
				# Forward propagation: compute the output
				hypothesis = self.qr_model(Xt)

				# Computation of the loss
				loss = self.pinball_loss(hypothesis, Tt)

				val_hypothesis = self.qr_model(Xt_val)
				val_loss = self.pinball_loss(val_hypothesis, Tt_val)

				# Backward propagation
				loss.backward() # <= compute the gradients
				
				# Update parameters (weights and biais)
				optimizer.step()

				# Print some performance to monitor the training
				tmp_loss.append(loss.item()) 
				tmp_val_loss.append(val_loss.item())

				
			losses.append(np.mean(tmp_loss))
			val_losses.append(np.mean(tmp_val_loss))

		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses, label="train", color="blue")
		plt.plot(np.arange(n_epochs), val_losses, label="valid", color="green")
		plt.title("QR loss")
		plt.legend()
		plt.tight_layout()
		fig_loss.savefig(self.results_path+"/qr_losses.png")
		plt.close()


	def save_model(self):
		self.net_path = self.results_path+"/qr_{}epochs.pt".format(self.n_epochs)
		torch.save(self.qr_model, self.net_path)


	def load_model(self, n_epochs):
		self.net_path = self.results_path+"/qr_{}epochs.pt".format(n_epochs)
		self.qr_model = torch.load(self.net_path)
		self.qr_model.eval()
		if cuda:
			self.qr_model.cuda()

