import numpy as np
from numpy.random import rand
import scipy.special
import copy
import torch
from torch.autograd import Variable
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class Classification_CP():
	

	def __init__(self, Xc, Yc, trained_model):
		self.Xc = Xc 
		self.Yc = Yc 
		self.labels = [0,1,2]
		self.trained_model = trained_model
		self.q = len(Yc) # number of points in the calibration set


	def initialize(self):
		self.cal_pred_output = self.trained_model(Variable(FloatTensor(self.Xc)))
		self.cal_pred_lkh = torch.nn.Softmax(dim=1)(self.cal_pred_output).cpu().detach().numpy()
		self.calibr_scores = self.get_nonconformity_scores(self.Yc, self.cal_pred_lkh) # nonconformity scores on the calibration set
		


	def get_nonconformity_scores(self, y, pred_lkh, sorting = True):

		n_points = len(y)
		ncm = np.array([np.abs(1-pred_lkh[i,int(y[i])]) for i in range(n_points)])
		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
		return ncm


	def get_p_values(self, x):
		'''
		calibr_scores: non conformity measures computed on the calibration set and sorted in descending order
		x: new input points (shape: (n_points,x_dim)
		
		return: positive p-values, negative p-values
		
		'''
		xt = Variable(FloatTensor(x))
		pred_output = self.trained_model(xt) # prob of going to pos class on x
		pred_lkh = torch.nn.Softmax(dim=1)(pred_output).cpu().detach().numpy()
		
		alphas = self.calibr_scores
		q = self.q
		
		n_points = len(pred_lkh)

		A0 = self.get_nonconformity_scores(self.labels[0]*np.ones(n_points), pred_lkh, sorting = False) # calibr scores for positive class
		A1 = self.get_nonconformity_scores(self.labels[1]*np.ones(n_points), pred_lkh, sorting = False) # calibr scores for positive class
		A2 = self.get_nonconformity_scores(self.labels[2]*np.ones(n_points), pred_lkh, sorting = False) # calibr scores for positive class
		
		p0 = np.zeros(n_points) # p-value for class 0
		p1 = np.zeros(n_points) # p-value for class 1
		p2 = np.zeros(n_points) # p-value for class 2
		
		for k in range(n_points):
			c0_a, c0_b, c1_a, c1_b, c2_a, c2_b = 0, 0, 0, 0, 0, 0
			for count_0 in range(self.q):
				if self.calibr_scores[count_0] > A0[k]:
					c0_a += 1
				elif self.calibr_scores[count_0] == A0[k]:
					c0_b += 1
				else:
					break
			for count_1 in range(self.q):
				if self.calibr_scores[count_1] > A1[k]:
					c1_a += 1
				elif self.calibr_scores[count_1] == A1[k]:
					c1_b += 1
				else:
					break
			for count_2 in range(self.q):
				if self.calibr_scores[count_2] > A2[k]:
					c2_a += 1
				elif self.calibr_scores[count_2] == A2[k]:
					c2_b += 1
				else:
					break
			p0[k] = ( c0_a + rand() * (c0_b + 1) ) / (self.q + 1)
			p1[k] = ( c1_a + rand() * (c1_b + 1) ) / (self.q + 1)
			p2[k] = ( c2_a + rand() * (c2_b + 1) ) / (self.q + 1)
		

		return p0, p1, p2


	def get_prediction_region(self, epsilon, p0, p1, p2):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		#		epsilon = confidence_level
		# OUTPUT: one-hot encoding of the prediction region [shape: (n_points,2)]
		# 		first column: negative class
		# 		second column: positive class
		n_points = len(p0)

		pred_region = np.zeros((n_points,3)) 
		for i in range(n_points):
			if p0[i] > epsilon:
				pred_region[i,0] = 1
			if p1[i] > epsilon:
				pred_region[i,1] = 1
			if p2[i] > epsilon:
				pred_region[i,2] = 1

		return pred_region

	def compute_prediction_region(self, x, eps):

		p0, p1, p2 = self.get_p_values(x)

		return self.get_prediction_region(eps, p0, p1, p2)


	def get_coverage(self, pred_region, labels):

		n_points = len(labels)

		c = 0
		for i in range(n_points):
			if pred_region[i,int(labels[i])] == 1:
				c += 1

		coverage = c/n_points

		return coverage


	def compute_coverage(self, eps, inputs, outputs):
		p0, p1, p2 = self.get_p_values(x = inputs)

		self.pred_region = self.get_prediction_region(eps, p0, p1, p2)

		return self.get_coverage(self.pred_region, outputs)



	def compute_efficiency(self):

		n_singletons = 0
		n_points = self.pred_region.shape[0]
		for i in range(n_points):
			if np.sum(self.pred_region[i]) == 1:
				n_singletons += 1

		return n_singletons/n_points

	def get_efficiency(self, pred_region):

		n_singletons = 0
		n_points = pred_region.shape[0]
		for i in range(n_points):
			if np.sum(pred_region[i]) == 1:
				n_singletons += 1

		return n_singletons/n_points