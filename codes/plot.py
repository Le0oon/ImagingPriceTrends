""" Plot loss and accuracy curve """

import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_acc(loss_and_acc_dict_train,loss_and_acc_dict_val, path):
	fig = plt.figure()
	tmp_train = list(loss_and_acc_dict_train.values())
	tmp_val = list(loss_and_acc_dict_val.values())
	maxEpoch = len(tmp_train[0][0])
	stride = np.ceil(maxEpoch / 10)

	maxLoss_train = max([max(x[0]) for x in loss_and_acc_dict_train.values()]) + 0.1
	maxLoss_val = max([max(x[0]) for x in loss_and_acc_dict_train.values()]) + 0.1
	maxLoss = max(maxLoss_train,maxLoss_val)
	minLoss_train = max(0, min([min(x[0]) for x in loss_and_acc_dict_train.values()]) - 0.1)
	minloss_val = max(0, min([min(x[0]) for x in loss_and_acc_dict_val.values()]) - 0.1)
	minLoss = min(minLoss_train,minloss_val)

	for name, lossAndAcc in loss_and_acc_dict_train.items():
		plt.plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)
	
	for name, lossAndAcc in loss_and_acc_dict_val.items():
		plt.plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.xticks(range(0, maxEpoch + 1, 2))
	plt.axis([0, maxEpoch, minLoss, maxLoss])
	plt.savefig(path[0])
	# plt.show()


	maxAcc_train = min(1, max([max(x[1]) for x in loss_and_acc_dict_train.values()]) + 0.1)
	maxAcc_val = min(1, max([max(x[1]) for x in loss_and_acc_dict_val.values()]) + 0.1)
	maxAcc = max(maxAcc_train,maxAcc_val)
	minAcc_train = max(0, min([min(x[1]) for x in loss_and_acc_dict_train.values()]) - 0.1)
	minAcc_val = max(0, min([min(x[1]) for x in loss_and_acc_dict_val.values()]) - 0.1)
	minAcc = min(minAcc_train,minAcc_val)
 
	fig = plt.figure()

	for name, lossAndAcc in loss_and_acc_dict_train.items():
		plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)
	for name, lossAndAcc in loss_and_acc_dict_val.items():
		plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.xticks(range(0, maxEpoch + 1, 2))
	plt.axis([0, maxEpoch, minAcc, maxAcc])
	plt.legend()
	# plt.show()
	plt.savefig(path[1])


'''
if __name__ == '__main__':
	loss = [x for x in range(10, 0, -1)]
	acc = [x / 10. for x in range(0, 10)]
	plot_loss_and_acc({'as': [loss, acc]})
'''

if __name__ == '__main__':
    loss = [x for x in range(10, 0, -1)]
    acc = [x / 10. for x in range(0, 10)]
    
    # Sample data for validation (You can adjust this according to your needs)
    val_loss = [x for x in range(10, 0, -1)]
    val_acc = [x / 10. for x in range(0, 10)]
    
    plot_loss_and_acc({'model_train': [loss, acc]}, {'model_val': [val_loss, val_acc]})