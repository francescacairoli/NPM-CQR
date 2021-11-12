import pickle
import numpy as np


class DetermDataset():

	def __init__(self, trainset_fn, testset_fn, calibrset_fn, alpha = 0.1, n_train_states = 2000, n_cal_states = 1000, n_test_states = 100, multiplier = 1):
		self.trainset_fn = trainset_fn
		self.testset_fn = testset_fn
		self.calibrset_fn = calibrset_fn
		self.n_train_states = n_train_states
		self.n_cal_states = n_cal_states
		self.n_test_states = n_test_states
		self.alpha = alpha
		self.multiplier = multiplier
		
		
	def load_data(self):
		self.load_train_data()
		self.load_test_data()
		self.load_calibration_data()
		
	def load_train_data(self):

		file = open(self.trainset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.X_train = data["x_scaled"]
		self.R_train = self.multiplier*data["rob"]
		
		self.x_dim = self.X_train.shape[1]
		self.n_training_points = self.X_train.shape[0]
		
		
		L_train = np.zeros((self.n_train_states,3)) #{-1,0,1}
		C_train = np.zeros(self.n_train_states)
		for i in range(self.n_train_states):

			if self.R_train[i] > 0:
				L_train[i, 2] = 1
				C_train[i] = 2
			elif self.R_train[i] < 0:
				L_train[i, 0] = 1
				C_train[i] = 0
			else:
				L_train[i, 1] = 1
				C_train[i] = 1

		self.L_train = L_train
		self.C_train = C_train
			
		
	def load_test_data(self):

		file = open(self.testset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.X_test = data["x_scaled"]
		self.R_test = self.multiplier*data["rob"]
		
		self.n_test_points = self.X_test.shape[0]
		
		
		L_test = np.zeros((self.n_test_states,3)) #{-1,0,1}
		C_test = np.empty(self.n_test_states)
		for i in range(self.n_test_states):
			
			if self.R_test[i] > 0:
				L_test[i, 2] = 1
				C_test[i] = 2
			elif self.R_test[i] < 0:
				L_test[i, 0] = 1
				C_test[i] = 0
			else:
				L_test[i, 1] = 1
				C_test[i] = 1
		self.L_test = L_test
		self.C_test = C_test

	def load_calibration_data(self):

		file = open(self.calibrset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.X_cal = data["x_scaled"]
		self.R_cal = self.multiplier*data["rob"]

		self.n_cal_points = self.X_cal.shape[0]

		L_cal = np.zeros((self.n_cal_states,3)) #{-1,0,1}
		C_cal = np.empty(self.n_cal_states)
		for i in range(self.n_cal_states):

			if self.R_cal[i] > 0:
				L_cal[i, 2] = 1
				C_cal[i] = 2 
			elif self.R_cal[i] < 0:
				L_cal[i, 0] = 1
				C_cal[i] = 0
			else:
				L_cal[i, 1] = 1
				C_cal[i] = 1
		self.L_cal = L_cal
		self.C_cal = C_cal
		
	def generate_mini_batches(self, n_samples):
		
		n_trajs = len(self.R_train)//len(self.X_train)
		
		ix = np.random.randint(0, len(self.R_train), n_samples)
		Xb = self.X_train[ix]
		Rb = self.R_train[ix]
		Lb = self.L_train[ix]
		Cb = self.C_train[ix]

		return Xb, Rb, Cb, ix
