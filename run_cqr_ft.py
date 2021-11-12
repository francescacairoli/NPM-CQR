from QR import *
from CQR import *
from RobSign import *
from TrainQR_multiquantile import *
from TrainRobSign import *
import copy
from Dataset import *
from CP_classification import *
from utils import *

model_name = "Sinusoidal"
print("MODEL = ", model_name)

trainset_fn, calibrset_fn, testset_fn, ds_details = import_filenames(model_name)
n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size = ds_details
print(trainset_fn)
print(testset_fn)
print(calibrset_fn)

n_epochs = 200
n_finetuning_epochs = 100

batch_size = 512
lr = 0.1
lr_finetuning = 0.1

qr_training_flag = True
sign_training_flag = True

xavier_flag = False
n_hidden = 10
opt = "Adam"
scheduler_flag = False

print("qr_training_flag = ", qr_training_flag)
print("sign_training_flag = ", sign_training_flag)

eps_regr = 0.027
quantiles = np.array([eps_regr/2,  1-eps_regr/2])#0.05, 0.5, 0.95,
nb_quantiles = len(quantiles)
#alpha = quantiles[0]*2
alpha = 0.1

print("Quantiles = ", quantiles)

print("n_epochs = {}, n_finetuning_epochs = {}, lr = {}, lr_ft = {}, batch_size = {}".format(n_epochs, n_finetuning_epochs, lr, lr_finetuning, batch_size))

idx = 'CQR+FT_#3_multiout_opt='+opt+'_w_Scheduler={}+XavierInit={}_{}hidden_#1_{}+{}epochs_{}quantiles_3layers_alpha01_eta0025'.format(scheduler_flag, xavier_flag, n_hidden,n_epochs, n_finetuning_epochs, nb_quantiles)

print("idx = {}".format(idx))
dataset = Dataset(trainset_fn, testset_fn, calibrset_fn, alpha, n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size)
dataset.load_data()

# Train the QR
qr = TrainQR(model_name, dataset, idx = idx, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = opt, n_hidden = n_hidden, xavier_flag = xavier_flag, scheduler_flag = scheduler_flag)
qr.initialize()

if qr_training_flag:
	qr.train(n_epochs, batch_size, lr)
	qr.save_model()
else:
	qr.load_model(n_epochs)

# Obtain CQR intervals
cqr = CQR(dataset.X_cal, dataset.R_cal, qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
cpi_test, pi_test = cqr.get_cpi(dataset.X_test, pi_flag = True)

cqr.plot_results(dataset.R_test, pi_test, "QR_interval", qr.results_path)
pi_coverage, pi_efficiency = cqr.get_coverage_efficiency(dataset.R_test, pi_test)
print("pi_coverage = ", pi_coverage, "pi_efficiency = ", pi_efficiency)
pi_correct, pi_uncertain, pi_wrong = cqr.compute_accuracy_and_uncertainty(pi_test, dataset.L_test)
print("pi_correct = ", pi_correct, "pi_uncertain = ", pi_uncertain, "pi_wrong = ", pi_wrong)

cqr.plot_results(dataset.R_test, cpi_test, "CQR_interval", qr.results_path)
cpi_coverage, cpi_efficiency = cqr.get_coverage_efficiency(dataset.R_test, cpi_test)
print("cpi_coverage = ", cpi_coverage, "cpi_efficiency = ", cpi_efficiency)
cpi_correct, cpi_uncertain, cpi_wrong = cqr.compute_accuracy_and_uncertainty(cpi_test, dataset.L_test)
print("cpi_correct = ", cpi_correct, "cpi_uncertain = ", cpi_uncertain, "cpi_wrong = ", cpi_wrong)

robsign = TrainRobSign(model_name, dataset, idx = idx, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, n_hidden = n_hidden, scheduler_flag = scheduler_flag)
robsign.initialize()
'''
# initialize the SignNN with the weights of the QR NN
sign_weights_list = [robsign.sign_model.fc_in.weight, robsign.sign_model.fc_1.weight, robsign.sign_model.fc_2.weight, robsign.sign_model.fc_3.weight, robsign.sign_model.fc_out.weight]
sign_biases_list = [robsign.sign_model.fc_in.bias, robsign.sign_model.fc_1.bias, robsign.sign_model.fc_2.bias, robsign.sign_model.fc_3.bias, robsign.sign_model.fc_out.bias]

qr_weights_list = [qr.qr_model.fc_in.weight, qr.qr_model.fc_1.weight, qr.qr_model.fc_2.weight, qr.qr_model.fc_3.weight, qr.qr_model.fc_out.weight]
qr_biases_list = [qr.qr_model.fc_in.bias, qr.qr_model.fc_1.bias, qr.qr_model.fc_2.bias, qr.qr_model.fc_3.bias, qr.qr_model.fc_out.bias]

for d in range(len(sign_weights_list)):
	sign_weights_list[d] = qr_weights_list[d] 
	sign_biases_list[d] = qr_biases_list[d]
'''
# Train the SignNN (FineTuning)
if qr_training_flag:
	robsign.train(n_finetuning_epochs, batch_size, lr_finetuning)
	robsign.save_model()
else:
	robsign.load_model(n_finetuning_epochs)

pred_test_sign = robsign.sign_model(Variable(FloatTensor(dataset.X_test))).cpu().detach().numpy()
pred_test_class = np.argmax(pred_test_sign, axis = 1)

test_accuracy = robsign.compute_accuracy(dataset.C_test, pred_test_sign)
print("Test sign classification accuracy = ", test_accuracy)

cpi_ft_test = copy.deepcopy(cpi_test)
for i in range(dataset.n_test_points):
	if pred_test_class[i] == 0 and cpi_ft_test[i,1] > 0: #-1
		cpi_ft_test[i,1] = 0

	if pred_test_class[i] == 2 and cpi_ft_test[i,0] < 0: #+1
		cpi_ft_test[i,0] = 0

cqr.plot_results(dataset.R_test, cpi_ft_test, "CQR_interval_with_finetuning", robsign.results_path)

cpi_ft_coverage, cpi_ft_efficiency = cqr.get_coverage_efficiency(dataset.R_test, cpi_ft_test)
print("FT cpi_coverage = ", cpi_ft_coverage, "cpi_efficiency = ", cpi_ft_efficiency)
cpi_ft_correct, cpi_ft_uncertain, cpi_ft_wrong = cqr.compute_accuracy_and_uncertainty(cpi_ft_test, dataset.L_test)
print("FT cpi_correct = ", cpi_ft_correct, "cpi_uncertain = ", cpi_ft_uncertain, "cpi_wrong = ", cpi_ft_wrong)

eta = 0.025
epsilon_class = alpha - eta
cp = Classification_CP(dataset.X_cal, dataset.C_cal, robsign.sign_model)
cp.initialize()

class_pred_region = cp.compute_prediction_region(dataset.X_test, epsilon_class)

cp_class_coverage = cp.get_coverage(class_pred_region, dataset.C_test)
cp_class_efficiency = cp.get_efficiency(class_pred_region)
print("cp_class_coverage = ", cp_class_coverage, "cp_class_efficiency = ", cp_class_efficiency)

cpi_cpft_test = copy.deepcopy(cpi_test)
for i in range(dataset.n_test_points):
	cpr_i = class_pred_region[i]
	if cpr_i[0] == 1 and np.sum(cpr_i) == 1 and cpi_cpft_test[i,1] > 0: #-1
		cpi_cpft_test[i,1] = 0

	if cpr_i[2] == 1 and np.sum(cpr_i) == 1 and cpi_cpft_test[i,0] < 0: #+1
		cpi_cpft_test[i,0] = 0

cqr.plot_results(dataset.R_test, cpi_cpft_test, "CQR_interval_with_CP_finetuning", robsign.results_path)

cpi_cpft_coverage, cpi_cpft_efficiency = cqr.get_coverage_efficiency(dataset.R_test, cpi_cpft_test)
print("CP-FT cpi_coverage = ", cpi_cpft_coverage, "cpi_efficiency = ", cpi_cpft_efficiency)
cpi_cpft_correct, cpi_cpft_uncertain, cpi_cpft_wrong = cqr.compute_accuracy_and_uncertainty(cpi_cpft_test, dataset.L_test)
print("CP-FT cpi_correct = ", cpi_cpft_correct, "cpi_uncertain = ", cpi_cpft_uncertain, "cpi_wrong = ", cpi_cpft_wrong)

