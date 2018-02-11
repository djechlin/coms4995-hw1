from hw1 import *
from helpers import *
import numpy as np

# Load the data
data_root_path = '/home/daniel/hw1/cifar10-hw1/'
# data_root_path = './cifar10-hw1/'
X_train, y_train = get_train_data(data_root_path, limit=50000) # this may take a few minutes

X_centered = (X_train - np.mean(X_train, axis=1)[:, np.newaxis])
X_div = (X_centered / np.std(X_centered, axis=1)[:, np.newaxis])

X_train = X_div # :)
# confirm
avgs = X_train.mean(axis = 1)
stds = X_train.std(axis = 1)

print("Normalized X_train to N(0,1): avgs=[%f, %f], stds=[%f, %f]" % (avgs.min(), avgs.max(), stds.min(), stds.max()))

X_test = get_images(data_root_path + 'test', limit=50000)
print('Data loading done')

# Part 1

alpha = 0.0005
batch = 150
layer_dimensions = [3072, 600, 10]
drop = 0.1
reg = 0.0000001
#layer_dimensions = [X_train.shape[0], some_ints[0], some_ints[1], some_ints[2], some_ints[3],  10]  # including the input and output layers
NN = NeuralNetwork(layer_dimensions, reg_lambda_1=reg, drop_prob=drop)#, optimizer="sgd_momentum")
print("Train: batch=%d, alpha=%f, layer=%s, reg=%f, drop=%f" % (batch, alpha, str(layer_dimensions), reg, drop))
NN.train(X_train, y_train, print_every=20, iters=1000, alpha=alpha, batch_size=batch)

# NN.train(X_train, y_train,
# 	iters=1000,
# 	alpha=0.1,
# 	batch_size=10,
# 	print_every=50)

y_predicted = NN.predict(X_test)
save_predictions('ans1-uni', y_predicted)

# test if your numpy file has been saved correctly
loaded_y = np.load('ans1-uni.npy')
print(loaded_y.shape)
loaded_y[:10]
