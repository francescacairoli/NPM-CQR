import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pickle
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries

class CoupledVanDerPol(object):

	def __init__(self, horizon = 8, formula = '', dt = 0.01):
		self.ranges = np.array([[-2.5, 2.5],[-4.05,4.05],[-2.5, 2.5],[-4.05,4.05]])
		self.horizon = horizon
		self.state_dim = 4
		self.obs_dim = 2
		self.phi = formula
		self.timeline = np.arange(start=0, stop=horizon, step=dt)
		self.dt = dt


	def set_goal(self, formula):
		self.phi = formula

	def diff_eq(self, z):
		# z = (x1,y1,x2,y2)
		
		mu = 1
		dzdt = np.array([
			z[1],
			mu*(1-z[0]**2)*z[1]-2*z[0]+z[2],
			z[3],
			mu*(1-z[2]**2)*z[3]-2*z[2]+z[0]
			])

		next_z = z+dzdt*self.dt
		return next_z

	def gen_rnd_states(self, n_samples):

		states = np.empty((n_samples, self.state_dim))
		for i in range(n_samples):
			#print("Point {}/{}".format(i+1,n_samples))
			
			states[i] = self.ranges[:,0]+(self.ranges[:,1]-self.ranges[:,0])*np.random.rand(self.state_dim)

		return states

	def gen_trajectories(self, states):

		n_samples = len(states)
		n_steps = len(self.timeline)

		trajs = np.empty((n_samples, n_steps, self.state_dim))
		
		for i in range(n_samples):
			trajs[i,0] = states[i]

			for t in range(1, n_steps):
				trajs[i, t] = self.diff_eq(trajs[i, t-1])
			
		return trajs


	def compute_robustness(self, trajs):
		n_states = len(trajs)
		robs = np.empty(n_states)
		 
		for i in range(n_states):
			print("rob ", i, "/", n_states)
			time_series_i = TimeSeries(['X1', 'Y1', 'X2', 'Y2'], self.timeline, trajs[i].T)
			robs[i] = stlRobustSemantics(time_series_i, 0, self.phi)

		return robs


if __name__=='__main__':

	n_points = 10000
	horizon = 10

	orig_threshold = 2.75

	cvdp_model = CoupledVanDerPol(horizon = horizon)
	states = cvdp_model.gen_rnd_states(n_points)
	future_trajs = cvdp_model.gen_trajectories(states)
	
	xmax = np.max(np.max(future_trajs, axis = 0), axis = 0)
	xmin = np.min(np.min(future_trajs, axis = 0), axis = 0)
	
	future_trajs_scaled = -1+2*(future_trajs-xmin)/(xmax-xmin)
	states_scaled = -1+2*(states-xmin)/(xmax-xmin)
	
	scaled_thresh_1 = -1+2*(orig_threshold-xmin[1])/(xmax[1]-xmin[1])
	scaled_thresh_2 = -1+2*(orig_threshold-xmin[3])/(xmax[3]-xmin[3])

	goal_formula_scaled = '( G_[0,{}] ( (Y1 < {}) & (Y2 < {}) ) )'.format(horizon, scaled_thresh_1, scaled_thresh_2)
	cvdp_model.set_goal(goal_formula_scaled)

	robs = cvdp_model.compute_robustness(future_trajs_scaled)
	print("Percentage of positive points: ", np.sum((robs>0))/n_points)

	dataset_dict = {"x": states, "trajs": future_trajs, "x_scaled": states_scaled, "trajs_scaled": future_trajs_scaled, "rob": robs}

	filename = 'Datasets/DetCVDP_calibration_set_{}points.pickle'.format(n_points)
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)
