from data import *
from autograd import numpy as np
from autograd import grad
import os
import shutil
import sys

# gradient descent function 
def gradient_descent(g,alpha,max_its,w,x,y):
	# compute gradient module using autograd
	gradient = grad(g)

	# run the gradient descent loop
	weight_history = [w] # weight history container
	cost_history = [g(w,x,y)] # cost function history container
	for k in range(max_its):
		# evaluate the gradient
		grad_eval = gradient(w,x,y)
		grad_eval /= np.linalg.norm(grad_eval)
#         grad_eval = np.sign(grad_eval)

		# take gradient descent step
		w = w - alpha*grad_eval
		
		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w,x,y))
	return weight_history,cost_history

# the import statement for matplotlib
import matplotlib.pyplot as plt

# cost function history plotter
def plot_cost_histories(cost_histories,labels):
	# create figure
	plt.figure()
	
	# loop over cost histories and plot each one
	for j in range(len(cost_histories)):
		history = cost_histories[j]
		label = labels[j]
		plt.plot(history,label = label)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show()

def model(w,x):
	return (w[0] + np.dot(x.T,w[1:])).T

def softmax(w,x,yj):
	cost = np.sum(np.log(np.ones(yj.size) + np.exp(-yj * model(w,x))))
	return cost/yj.size

def perceptron(w,x,yj):
	cost = np.sum(np.maximum(np.zeros((yj.shape)), (-1 * y * model(w, x))))
	return cost / yj.size


def feature_extraction(num):
	# subset_dir = './3/subsets/'
	# subset_list = os.listdir(subset_dir)

	positive_class_num = num
	positive_class_path = subset_dir + subset_list[positive_class_num]

	X,Y,used_tokens = form_dataset(subset_dir,positive_class_num)

	dimx = X.shape[0]
	# dimy = Y.shape[0]

	w = np.random.rand(dimx + 1,1)
	alpha = 3 * 10 ** (-2)
	max_its = 500
	weight_history, cost_history = gradient_descent(softmax,alpha,max_its,w,X,Y)

	cost_histories = [cost_history]
	labels = ['test']
	# plot_cost_histories(cost_histories,labels)

	final_loss = cost_history[-1]
	final_weight = weight_history[-1]
	# final_weight = np.array(final_weight)

	print ('Final loss: ', final_loss)
	# print ('Final weights: ', final_weight)

	positive_tokens = get_tokens(positive_class_path)
	positive_tokens_num = len(positive_tokens)

	top_num = 15

	weights = []
	for i,w in enumerate(positive_tokens):
		if w in used_tokens:
			index = used_tokens.index(w)
			weights.append(final_weight[index])

	scores = copy.copy(weights)
	positive_tokens_b = copy.copy(positive_tokens)

	print ('\n' + 'Top {} features:'.format(top_num))

	features_extracted = []
	for i in range(top_num):
		highest = max(scores)
		highest_index = scores.index(highest)

		print (positive_tokens_b[highest_index],highest[0])
		features_extracted.append([positive_tokens_b[highest_index], highest])

		del scores[highest_index]
		del positive_tokens_b[highest_index]

	return features_extracted

feature_output_dir = './data/features/'
if not os.path.exists(feature_output_dir):
	os.mkdir(feature_output_dir)

subset_dir = './data/subsets/'
subset_list = os.listdir(subset_dir)

# batch operation
'''
for i,subset in enumerate(subset_list):
	subset_name = subset[:subset.rfind('.')]
	f = open(feature_output_dir + subset_name + '.txt', 'w')
	features = feature_extraction(i)
	for fea in features:
		f.write(str(fea[0]) + '\r\n')
	print ('\n', subset_name, ' :)')
'''

# one by one operation
i = int(sys.argv[1])
subset = subset_list[i]
subset_name = subset[:subset.rfind('.')]
f = open(feature_output_dir + subset_name + '.txt', 'w')
features = feature_extraction(i)
for fea in features:
	f.write(fea[0] + ' '+ str(fea[1][0]) + '\r\n')
print ('\n', subset_name, ' :)')
