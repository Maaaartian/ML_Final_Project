import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path
from generator import data_generator
import scipy.linalg as la


def Glasso(A, lambda_=0.005, iter=100):
	# row: features, column: samples
	m = A.shape[1]
	k = A.shape[0]
	S = np.cov(A)
	# W: covariance matrix, T: precision matrix (W's inverse)
	W = S.copy() + 0.001 * np.eye(k) # how to choose the lambda value?
	T = np.linalg.inv(W)
	# take the absolute of all negative values in the covariance matrix
	'''
	print('Before:')
	print(W)
	'''
	indices = np.arange(k)
	for n in range(iter):
		for i in range(k):
			# upper left sub-matrix of W
			W_11 = W[indices != i].T[indices != i]
			S_12 = W[indices != i, i]
			#solve the lasso problem
			
			_, _, coefs_ = lars_path( W_11, S_12, Xy = S_12, Gram = W_11,
										alpha_min = lambda_/(k-1), copy_Gram = True,
										method = "lasso")
			coefs_ = coefs_[:,-1] # only the last column
			'''
			X = la.sqrtm(W_11)
			y = np.dot(np.linalg.inv(X), S_12)
			coefs_ = [0 for j in range(k-1)]
			coefs_ = lasso_regression(X, y, coefs_, lambda_)
			'''
			#update the precision matrix.
			T[i, i] = 1 / (W[i,i] - np.dot(W[indices != i, i], coefs_))
			T[indices != i, i] = - T[i, i] * np.asarray(coefs_)
			T[i, indices != i] = - T[i, i] * np.asarray(coefs_)
			temp_coefs = np.dot(W_11, coefs_)
			W[i, indices != i] = temp_coefs
			W[indices != i, i] = temp_coefs
		'''
		print('Iter:', n+1)
		print(W)
		'''
		# test convergence
		if np.abs(dual_gap(W, T, lambda_)) < 0.1: # should S be W here?
			break
	'''
	print('After:')
	print(W)
	'''
	graph = build_graph(W)
	return graph


def lasso_regression(X, y, weights, lambda_, iter=300):
	# k = number of coordinates
	k = len(weights)
	for n in range(iter):
		for i in range(k):
			ceiling = -np.dot(X[:, i], y-np.dot(X, weights))
			floor = np.dot(X[:, i], X[:, i])
			upper = (ceiling + lambda_/2) / floor
			lower = (ceiling - lambda_/2) / floor

			if weights[i] > upper:
				weights[i] -= upper
			elif weights[i] < lower:
				weights -= lower
			else:
				weights[i] = 0
	return weights


def dual_gap(emp_cov, precision_, alpha):
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return gap 


def find_negative_values(matrix):
	rows = matrix.shape[0]
	columns = matrix.shape[1]
	negative_positions = []
	for i in range(rows):
		for j in range(columns):
			if matrix[i, j] < 0:
				matrix[i, j] = -matrix[i, j]
				negative_positions.append((i, j))
	return matrix, negative_positions


def recover_negative_values(matrix, negative_positions):
	for pos in negative_positions:
		matrix[pos[0], pos[1]] = -matrix[pos[0], pos[1]]
	return matrix


def build_graph(matrix):
	k = matrix.shape[0]
	parameters = np.zeros((k, k))
	for i in range(k):
		for j in range(k):
			# parameters for each vertex (clique with size 1)
			if i == j:
				parameters[i, j] = np.sqrt(matrix[i, j])
			else:
				if matrix[i, j] > 0.01:
					parameters[i, j] = matrix[i, j]
	return parameters


def read_data(name):
	data = pd.read_csv(name)
	return data


if __name__ == '__main__':
	# read the data set
	data = read_data('example.csv')
	names = list(data.columns)
	data = data.values.T
	# using graphical lasso to learn the graph
	graph = Glasso(data)
	# save the graph to a csv file
	result = pd.DataFrame(graph, columns=names)
	result.to_csv('graph.csv', index=False)

	