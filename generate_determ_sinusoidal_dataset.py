import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

model_name = "DetSinusoidal"
x_domain = [0, 2*math.pi]

n_train_points = 20000
n_cal_points = n_train_points//2
n_test_points = 10000

# train set
x_train = np.linspace(x_domain[0], x_domain[1], n_train_points)
y_train = np.sin(x_train)

# calibration set
x_cal = np.linspace(x_domain[0], x_domain[1], n_cal_points)
y_cal = np.sin(x_cal)

# test set
x_test = np.linspace(x_domain[0], x_domain[1], n_test_points)
y_test = np.sin(x_test)

train_dict = {"x_scaled": np.expand_dims(x_train,axis=1), "rob": y_train}

train_filename = 'Datasets/DetSinusoidal_train_set_{}points.pickle'.format(n_train_points)
with open(train_filename, 'wb') as handle:
	pickle.dump(train_dict, handle)
handle.close()
print("Data stored in: ", train_filename)

cal_dict = {"x_scaled": np.expand_dims(x_cal,axis=1), "rob": y_cal}

cal_filename = 'Datasets/DetSinusoidal_calibration_set_{}points.pickle'.format(n_cal_points)
with open(cal_filename, 'wb') as handle:
	pickle.dump(cal_dict, handle)
handle.close()
print("Data stored in: ", cal_filename)

test_dict = {"x_scaled": np.expand_dims(x_test,axis=1), "rob": y_test}

test_filename = 'Datasets/DetSinusoidal_test_set_{}points.pickle'.format(n_test_points)
with open(test_filename, 'wb') as handle:
	pickle.dump(test_dict, handle)
handle.close()
print("Data stored in: ", test_filename)


