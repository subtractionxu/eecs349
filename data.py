'''
Xu, Jian 2018.11.18
form data set
'''
import os
from preprocess import *
import numpy as np
import copy
import random

def get_tokens(set_path):
	return tokenize_subset(set_path)


def get_subset_sizes(subset_dir):
	subset_list = os.listdir(subset_dir)
	subset_sizes = []
	for subset in subset_list:
		subset_path = subset_dir + subset
		subset_file = open(subset_path,'r')
		reader = csv.reader(subset_file)
		subset_size = sum(1 for line in reader)
		subset_sizes.append(subset_size - 1)
	return subset_sizes

def ramdom_neg_samples(class_num_pos, subset_sizes):
	original_csv = './data/mtsamples.csv'
	csv_file = open(original_csv,'r')
	csv_reader = csv.reader(csv_file)
	line_num = sum(1 for l in csv_reader)
	csv_file = open(original_csv,'r')
	csv_reader = csv.reader(csv_file)
	csv_reader = list(csv_reader)

	start_num = sum(subset_sizes[i] for i in range(class_num_pos)) + 1
	end_num = sum(subset_sizes[i] for i in range(class_num_pos + 1)) + 1

	negative_set_dir = './data/negative_smps/'
	if not os.path.exists(negative_set_dir):
		os.mkdir(negative_set_dir)

	negative_set_path = negative_set_dir + str(class_num_pos) + '.csv'
	if os.path.isfile(negative_set_path):
		return negative_set_path
	csv_file_neg = open(negative_set_path, 'w')
	csv_writer = csv.writer(csv_file_neg)
	csv_writer.writerow(csv_reader[0])

	neg_num = 0
	while neg_num != subset_sizes[class_num_pos]:
		ram_ind = random.randint(1,line_num -1)
		# print (ram_ind)
		if ram_ind >= start_num and ram_ind <= end_num:
			continue
		csv_writer.writerow(csv_reader[ram_ind])
		neg_num += 1
	csv_file_neg.close()

	return negative_set_path

def vectorize_samples(positive_set_path,negative_set_path,tokens,P1):
	N = len(tokens)
	P =  2 * P1
	X = np.zeros((N,P))
	Y = np.zeros(P)
	pos_file = open(positive_set_path,'r')
	pos_reader = csv.reader(pos_file)
	for i,line in enumerate(pos_reader):
		if i == 0:
			continue
		x_i = np.zeros(len(tokens))
		filtered_line = tokenize_line(line)
		for f in filtered_line:
			if f in tokens:
				j = tokens.index(f)
				x_i[j] += 1
		X[:,i - 1] = x_i
		# assign positive sample with label '1'
		Y[i - 1] = 1
	neg_file = open(negative_set_path,'r')
	neg_reader = csv.reader(neg_file)
	for i,line in enumerate(neg_reader):
		if i == 0:
			continue
		x_i = np.zeros(len(tokens))
		filtered_line = tokenize_line(line)
		for f in filtered_line:
			if f in tokens:
				j = tokens.index(f)
				x_i[j] += 1
		# print (i)
		X[:,P1 + i - 1] = x_i
		Y[P1 + i - 1] = 0
	return X,Y

def form_dataset(subset_dir,class_num):
	# get tokens from the positive class
	subset_list = os.listdir(subset_dir)
	class_num_pos = class_num
	positive_set_path = subset_dir + subset_list[class_num_pos]
	positive_tokens = get_tokens(positive_set_path)
	# print ('Positive class number: ', class_num_pos)
	print ('Positive class: ', subset_list[class_num_pos])

	# get tokens from the negative class
	subset_sizes = get_subset_sizes(subset_dir)
	negative_set_path = ramdom_neg_samples(class_num_pos,subset_sizes)
	nagative_tokens = get_tokens(negative_set_path)

	# combine the two and remove repeated tokens
	tokens = []
	tokens.extend(positive_tokens)
	tokens.extend(nagative_tokens)

	# remove the repeated tokens
	tokens = list(set(tokens))

	P1 = subset_sizes[class_num_pos]
	# P2 = subset_sizes[class_num_neg]
	X, Y = vectorize_samples(positive_set_path,negative_set_path,tokens,P1)

	return X,Y,tokens
