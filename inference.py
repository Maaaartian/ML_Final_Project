import numpy as np
import pandas as pd
from scipy import stats


class feature():

	def __init__(self, i):
		self.index = i
		self.parents = []
		self.children = []	
		self.all_values = []
		self.value = None
		self.elim_func = None
		self.mode = 0


	def add_parents(self, parent_list):
		self.parents = parent_list


	def add_child(self, child):
		self.children.append(child)


	def add_all_values(self, values):
		for i in values:
			if np.isnan(i):
				break
			self.all_values.append(i)


# compute elimination function for each feature
def compute_elim_func(x, graph_parameters, feature_list):
	i = x.index
	# the clique of the feature itself
	x.elim_func = phi((x.value,))
	for p in x.parents:
		# if the parent feature is also empty, use the mode value
		if feature_list[p].value == None:
			temp = feature_list[p].mode
		temp = feature_list[p].value
		j = feature_list[p].index
		# the clique of each edge between the feature and its parent
		x.elim_func *= graph_parameters[i, j] * phi((x.value, temp))
	# message passing - accumulate elimination functions from all children
	if x.children == []: # leaf node
		return
	for c in x.children:
		x.elim_func *= feature_list[c].elim_func


# predict the missing value
def predict(index, feature_list, order, graph_parameters):
	x = feature_list[index]
	# compute the elimiation function in the order of variable elimination
	print(f'Prediction for feature: X{index}')
	total_prob = []
	prob = []
	for v in x.all_values:
		x.value = v
		prob.append(v)
		for i in order:
			# compute elimination values for all children of the current feature
			if i == index:
				break
			compute_elim_func(feature_list[i], graph_parameters, feature_list)
		compute_elim_func(x, graph_parameters, feature_list)
		total_prob.append(x.elim_func)
	temp_list = []
	for i in range(len(prob)):
		print(f'P(X = {prob[i]}) = {total_prob[i] / sum(total_prob)}')
		temp_list.append(total_prob[i] / sum(total_prob))


# make predictions for all empty values for each sample
def inference(sample, graph_parameters, feature_list, order):
	# number of features
	k = len(sample)
	# assign values for each feature
	for i in range(k):
		feature_list[i].value = sample[i]
	# make predictions for each empty value
	for i in order:
		if np.isnan(feature_list[i].value):
			predict(i, feature_list, order, graph_parameters)


# potential function
def phi(var_list):
	# suppose the maximum size of the clique is 2
	if len(var_list) == 1:
		return np.exp(-np.square(var_list[0]))
	else:
		return np.exp(-np.square(np.abs(var_list[0]-var_list[1])))
	

# initialize all features (children, parents, all possible values)
def initialize_features(k, graph_dict, feature_values, feature_modes):
	feature_list = []
	# add all parent nodes for each feature
	for i in range(k):
		x = feature(i)
		x.add_parents(graph_dict[i])
		x.add_all_values(feature_values[:,i])
		x.mode = feature_modes[i]
		feature_list.append(x)
	# add all children nodes for each feature
	for i in range(len(feature_list)):
		for j in feature_list[i].parents:
			if not i in feature_list[j].parents:
				feature_list[j].add_child(i)
	return feature_list


if __name__ == '__main__':
	
	# structure of the graph
	graph_dict = {}
	graph_dict[0] = [9]
	graph_dict[1] = [8]
	graph_dict[2] = [4, 7, 8]
	graph_dict[3] = [7, 9]
	graph_dict[4] = [2, 7, 8]
	graph_dict[5] = [4, 7]
	graph_dict[6] = [5]
	graph_dict[7] = [2, 4, 8]
	graph_dict[8] = [2, 4, 7]
	graph_dict[9] = [7, 8]
	# order of elimination
	order = [0, 1, 6, 3, 5, 9, 2, 4, 7, 8]
	# parameters of the graph
	graph_parameters = pd.read_csv('test_parameters.csv').values
	# all possible values for each feature
	feature_values = pd.read_csv('test_values.csv').values

	data_set = pd.read_csv('test_query.csv').values
	# compute the mode value for each feature
	feature_modes = []
	# number of features
	k = graph_parameters.shape[0]
	for i in range(k):
		mode = stats.mode(data_set[:,i])[0][0]
		feature_modes.append(mode)
	# initializing features
	feature_list = initialize_features(k, graph_dict, feature_values, feature_modes)

	for i in data_set:
		inference(i, graph_parameters, feature_list, order)
		print('---------')
	