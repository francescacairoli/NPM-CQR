import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

model_name = "Sinusoidal"
x_domain = [0, 2*math.pi]
sigma = 0.125

n_train_points = 2000
train_hist_size = 50
n_cal_points = n_train_points//2
n_test_points = 200
test_hist_size = 2000

# train set
x_train = np.linspace(x_domain[0], x_domain[1], n_train_points)
y_train = []
for i in range(n_train_points):
	for _ in range(train_hist_size):
		y_train.append(np.sin(x_train[i])+np.random.normal(loc = 0, scale = sigma*np.abs(np.sin(x_train[i]))) )

# calibration set
x_cal = np.linspace(x_domain[0], x_domain[1], n_cal_points)
y_cal = []
for i in range(n_cal_points):
	for _ in range(train_hist_size):
		y_cal.append(np.sin(x_cal[i])+np.random.normal(loc = 0, scale = sigma*np.abs(np.sin(x_cal[i]))) )

# test set
x_test = np.linspace(x_domain[0], x_domain[1], n_test_points)
y_test = []
for i in range(n_test_points):
	for _ in range(test_hist_size):
		y_test.append(np.sin(x_test[i])+np.random.normal(loc = 0, scale = sigma*np.abs(np.sin(x_test[i]))) )

train_dict = {"x_scaled": np.expand_dims(x_train,axis=1), "rob": np.array(y_train)}

train_filename = 'Datasets/Sinusoidal_train_set_{}x{}points.pickle'.format(n_train_points, train_hist_size)
with open(train_filename, 'wb') as handle:
	pickle.dump(train_dict, handle)
handle.close()
print("Data stored in: ", train_filename)

cal_dict = {"x_scaled": np.expand_dims(x_cal,axis=1), "rob": np.array(y_cal)}

cal_filename = 'Datasets/Sinusoidal_calibration_set_{}x{}points.pickle'.format(n_cal_points, train_hist_size)
with open(cal_filename, 'wb') as handle:
	pickle.dump(cal_dict, handle)
handle.close()
print("Data stored in: ", cal_filename)

test_dict = {"x_scaled": np.expand_dims(x_test,axis=1), "rob": np.array(y_test)}

test_filename = 'Datasets/Sinusoidal_test_set_{}x{}points.pickle'.format(n_test_points, test_hist_size)
with open(test_filename, 'wb') as handle:
	pickle.dump(test_dict, handle)
handle.close()
print("Data stored in: ", test_filename)


x_train_rep = np.repeat(x_train, train_hist_size)
fig = plt.figure()
plt.scatter(x_train_rep, y_train, s=0.1)
plt.plot(x_train_rep, np.zeros(n_train_points*train_hist_size), '--', c='r')
plt.savefig(model_name+"/train_points.png")
plt.close()

x_cal_rep = np.repeat(x_cal, train_hist_size)
fig = plt.figure()
plt.scatter(x_cal_rep, y_cal, s=0.1)
plt.plot(x_cal_rep, np.zeros(n_cal_points*train_hist_size), '--', c='r')
plt.savefig(model_name+"/cal_points.png")
plt.close()

x_test_rep = np.repeat(x_test, test_hist_size)
fig = plt.figure()
plt.scatter(x_test_rep, y_test, s=0.1)
plt.plot(x_test_rep, np.zeros(n_test_points*test_hist_size), '--', c='r')
plt.savefig(model_name+"/test_points.png")
plt.close()