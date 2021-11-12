from RobSign import *
import numpy as np
import os 
import pickle
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class TrainDetermRobSign():

	def __init__(self, model_name, dataset, idx = None, quantiles = [0.05, 0.95], opt = "Adam", n_hidden = 50, scheduler_flag = False):

		self.model_name = model_name
		self.dataset = dataset
		
		if idx:
			self.idx = idx
			self.results_path = self.model_name+"/QR_results/ID_"+idx
		else:
			rnd_idx = str(np.random.randint(0,100000))
			self.idx = rnd_idx
			self.results_path = self.model_name+"/QR_results/ID_"+rnd_idx
		os.makedirs(self.results_path, exist_ok=True)

		self.valid_set_dim = 1000

		self.quantiles = quantiles
		self.nb_quantiles = len(quantiles)
		self.opt = opt
		self.n_hidden = n_hidden
		self.scheduler_flag = scheduler_flag


	def initialize(self):

		self.dataset.load_data()

		self.sign_model = RobSign(input_size = int(self.dataset.x_dim), hidden_size = self.n_hidden, output_size = self.nb_quantiles)

		if cuda:
			self.sign_model.cuda()

	def train(self, n_epochs, batch_size, lr):

		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.lr = lr

		class_loss_fnc = nn.CrossEntropyLoss()

		if self.opt == "Adam":
			optimizer = torch.optim.Adam(self.sign_model.parameters(), lr=lr)
		else:
			optimizer = torch.optim.RMSprop(self.sign_model.parameters(), lr=lr)
		scheduler = ExponentialLR(optimizer, gamma=0.9)


		self.net_path = self.results_path+"/robsign_{}epochs.pt".format(n_epochs)

		losses = []
		val_losses = []
		val_accuracies = []

		bat_per_epo = self.dataset.n_training_points // batch_size
		
		Xt_val = Variable(FloatTensor(self.dataset.X_cal[:self.valid_set_dim]))
		C_val = self.dataset.C_cal[:self.valid_set_dim]
		Ct_val = Variable(LongTensor(C_val))
				
		for epoch in range(n_epochs):
			
			if (epoch+1) % 25 == 0:
				print("Epoch= {},\t loss = {:2.4f}".format(epoch+1, losses[-1]))

			tmp_val_loss = []
			tmp_val_acc = []
			tmp_loss = []

			for i in range(bat_per_epo):
				# Select a minibatch
				state, rob, sign, b_ix = self.dataset.generate_mini_batches(batch_size)
				Xt = Variable(FloatTensor(state))
				Ct = Variable(LongTensor(sign))
				
				# initialization of the gradients
				optimizer.zero_grad()
				
				# Forward propagation: compute the output
				hypothesis = self.sign_model(Xt)

				# Computation of the loss
				loss = class_loss_fnc(hypothesis, Ct)

				val_hypothesis = self.sign_model(Xt_val)
				val_loss = class_loss_fnc(val_hypothesis, Ct_val)
				val_acc = self.compute_accuracy(C_val, val_hypothesis.cpu().detach().numpy())
				# Backward propagation
				loss.backward() # <= compute the gradients
				
				# Update parameters (weights and biais)
				optimizer.step()

				# Print some performance to monitor the training
				tmp_loss.append(loss.item()) 
				tmp_val_loss.append(val_loss.item())
				tmp_val_acc.append(val_acc)
			scheduler.step()
				
			losses.append(np.mean(tmp_loss))
			val_losses.append(np.mean(tmp_val_loss))
			val_accuracies.append(np.mean(tmp_val_acc))

		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses, label="train", color="blue")
		plt.plot(np.arange(n_epochs), val_losses, label="valid", color="green")
		plt.title("RobSign loss")
		plt.legend()
		plt.tight_layout()
		fig_loss.savefig(self.results_path+"/robsign_losses.png")
		plt.close()

		fig_acc = plt.figure()
		plt.plot(np.arange(n_epochs), val_accuracies, label="valid", color="green")
		plt.title("RobSign accuracy")
		plt.legend()
		plt.tight_layout()
		fig_acc.savefig(self.results_path+"/robsign_accuracies.png")
		plt.close()


	def compute_accuracy(self, C, C_pred):

		n_points = len(C)
		pred_class = np.argmax(C_pred, axis = 1)

		n_correct = 0
		for i in range(n_points):
			if C[i] == pred_class[i]:
				n_correct += 1

		return n_correct/n_points


	def save_model(self):
		self.net_path = self.results_path+"/robsign_{}epochs.pt".format(self.n_epochs)
		torch.save(self.sign_model, self.net_path)


	def load_model(self, n_epochs):
		self.net_path = self.results_path+"/robsign_{}epochs.pt".format(n_epochs)
		self.sign_model = torch.load(self.net_path)
		self.sign_model.eval()
		if cuda:
			self.sign_model.cuda()