
def import_determ_filenames(model_name):

	n_train_states = 20000
	n_cal_states = 10000
	n_test_states = 10000

	trainset_fn = "Datasets/"+model_name+"_train_set_{}points.pickle".format(n_train_states)
	testset_fn = "Datasets/"+model_name+"_test_set_{}points.pickle".format(n_test_states)
	calibrset_fn = "Datasets/"+model_name+"_calibration_set_{}points.pickle".format(n_cal_states)

	return trainset_fn, calibrset_fn, testset_fn, (n_train_states, n_cal_states, n_test_states)


def import_filenames(model_name):

	if model_name == "RT":
		n_train_states = 50000
		n_cal_states = 25000
		n_test_states = 100
		cal_hist_size = 1
		test_hist_size = 200

		trainset_fn = "Datasets/"+model_name+"_train_set_{}x{}points.pickle".format(n_train_states, cal_hist_size)
		testset_fn = "Datasets/"+model_name+"_test_set_{}x{}points.pickle".format(n_test_states, test_hist_size)
		calibrset_fn = "Datasets/"+model_name+"_calibration_set_{}x{}points.pickle".format(n_cal_states, cal_hist_size)

	elif model_name == "RT1" or model_name == "SIR1":
		n_train_states = 2000
		n_cal_states = 1000
		n_test_states = 500
		cal_hist_size = 50
		test_hist_size = 2000
		
		trainset_fn = "Datasets/"+model_name+"_train_set_{}x{}points.pickle".format(n_train_states, cal_hist_size)
		testset_fn = "Datasets/"+model_name+"_test_set_{}x{}points_grid.pickle".format(n_test_states, test_hist_size)
		calibrset_fn = "Datasets/"+model_name+"_calibration_set_{}x{}points.pickle".format(n_cal_states, cal_hist_size)
	elif model_name == "CVDP":
		n_train_states = 2000
		n_cal_states = 1000
		n_test_states = 100
		cal_hist_size = 50
		test_hist_size = 2000
		
		trainset_fn = "Datasets/"+model_name+"_train_set_{}x{}points.pickle".format(n_train_states, cal_hist_size)
		testset_fn = "Datasets/"+model_name+"_test_set_{}x{}points.pickle".format(n_test_states, test_hist_size)
		calibrset_fn = "Datasets/"+model_name+"_calibration_set_{}x{}points.pickle".format(n_cal_states, cal_hist_size)

	else:
		n_train_states = 2000
		n_cal_states = 1000
		n_test_states = 200
		cal_hist_size = 50
		test_hist_size = 2000
		
		trainset_fn = "Datasets/"+model_name+"_train_set_{}x{}points.pickle".format(n_train_states, cal_hist_size)
		testset_fn = "Datasets/"+model_name+"_test_set_{}x{}points.pickle".format(n_test_states, test_hist_size)
		calibrset_fn = "Datasets/"+model_name+"_calibration_set_{}x{}points.pickle".format(n_cal_states, cal_hist_size)

	return trainset_fn, calibrset_fn, testset_fn, (n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size)


def save_results_to_file(results_list, filepath):

	f = open(filepath+"/results.txt", "w")
	for i in range(len(results_list)):
		f.write(results_list[i])
	f.close()