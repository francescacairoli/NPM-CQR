import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
import math
import pickle
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries

class RoomTemperature(object):

	def __init__(self, horizon = 3, formula = ''):
		self.ranges = np.array([17, 30])
		self.horizon = horizon
		self.state_dim = 1
		self.phi = formula
		self.timeline = np.arange(horizon+1)


	def set_goal(self, formula):
		self.phi = formula

	def dynamics(self, x, t):
		
		ts = 5
		ae = 8*10**(-3)
		ah = 3.6*10**(-3)
		Te = 15
		Th = 55
		
		u = self.controller(t)
		next_x = x + ts*(ae*(Te-x)+ah*(Th-x)*u)

		return next_x

	def controller(self, t):

		u = -1.018*10**(-6)*t**4+7.563*10**(-5)*t**3-0.001872*t**2+0.02022*t+0.3944

		return u

	def gen_rnd_states(self, n_samples):

		states = np.empty((n_samples, self.state_dim))
		for i in range(n_samples):
			
			states[i] = self.ranges[0]+(self.ranges[1]-self.ranges[0])*np.random.rand()

		return states

	def gen_grid_states(self, n_samples):

		states = np.linspace(self.ranges[0], self.ranges[1], n_samples)
		states = states[:, np.newaxis]
		
		return states


	def gen_trajectories(self, states):

		n_samples = len(states)
		trajs = np.empty((n_samples, self.horizon+1, self.state_dim))
		
		for i in range(n_samples):
			prev_x = states[i]
			trajs[i, 0, 0] = states[i]
		
			for t in range(1, self.horizon+1):
				trajs[i, t, 0] = self.dynamics(prev_x, t)
				prev_x = trajs[i, t, 0] 
		return trajs


	def gen_bool_labels(self, trajs):

		n_states = len(trajs)
		labels = np.empty(n_states)
		
		for i in range(n_states):
			time_series_i = TimeSeries(['T'], self.timeline, trajs[i].T)
			labels[i] = stlBooleanSemantics(time_series_i, 0, self.phi)

		return labels

	def compute_robustness(self, trajs):
		n_states = len(trajs)
		robs = np.empty(n_states)
		 
		for i in range(n_states):
			time_series_i = TimeSeries(['T'], self.timeline, trajs[i].T)
			robs[i] = stlRobustSemantics(time_series_i, 0, self.phi)

		return robs


if __name__=='__main__':

	n_points = 10000
		
	horizon = 12 # next hour

	orig_lb = 17
	orig_ub = 28

	rt_model = RoomTemperature(horizon = horizon)
	#states = rt_model.gen_rnd_states(n_points)
	states = rt_model.gen_grid_states(n_points)
	
	future_trajs = rt_model.gen_trajectories(states)
	print("--- future_trajs.shape = ", future_trajs.shape)
	# scale trajs and states and thresholds
	xmax = np.max(np.max(future_trajs, axis = 0), axis = 0)
	xmin = np.min(np.min(future_trajs, axis = 0), axis = 0)
	print("--- xmin, xmax = ", xmin, xmax)
	future_trajs_scaled = -1+2*(future_trajs-xmin)/(xmax-xmin)
	states_scaled = -1+2*(states-xmin)/(xmax-xmin)

	scaled_lb = -1+2*(orig_lb-xmin[0])/(xmax[0]-xmin[0])
	scaled_ub = -1+2*(orig_ub-xmin[0])/(xmax[0]-xmin[0])
	
	goal_formula_scaled = '( G_[0,{}] ( (T <= {}) & (T >= {}) ) )'.format(horizon+1, scaled_ub, scaled_lb)
	rt_model.set_goal(goal_formula_scaled)
	
	robs = rt_model.compute_robustness(future_trajs_scaled)

	dataset_dict = {"x": states, "trajs": future_trajs, "rob": robs, 
					"x_scaled": states_scaled, "trajs_scaled": future_trajs_scaled,  "x_minmax": (xmin,xmax)}

	filename = 'Datasets/DetRT_calibration_set_{}points_grid.pickle'.format(n_points)
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)
