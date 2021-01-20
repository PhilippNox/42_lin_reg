import math
from collections import deque

MAX_ITER = 1000000
ERR_EPSILON = 0.001
WIN_SIZE = 4


def transpose_data(income):
	if len(income) == 0:
		return income

	data = []
	for _ in income[0]:
		data.append(list())

	for point in income:
		for idx in range(len(data)):
			data[idx].append(point[idx])
	return data


def means(matrix):
	out = []
	for row in matrix:
		out.append(sum(row) / len(row))
	return out


def sigmas(matrix, means=None):
	out = []
	for idx, row in enumerate(matrix):
		tmp_mean = sum(row) / len(row) if means is None else means[idx]
		tmp_units = map(lambda x: math.pow(x - tmp_mean, 2), row)
		out.append(math.sqrt(sum(tmp_units) / len(row)))
	return out


def standardization_vector(matrix):
	tmp_means = means(matrix)
	tmp_sigmas = sigmas(matrix, tmp_means)
	return list(zip(tmp_means, tmp_sigmas))


def standardization_apply(vector, param):
	out = []
	for idx, elem in enumerate(vector):
		out.append((elem - param[idx][0]) / param[idx][1])
	return out


def standardization(matrix):
	std_vector = standardization_vector(matrix)
	out = []
	for idx, row in enumerate(matrix):
		out.append(list(map(lambda x: (x - std_vector[idx][0]) / std_vector[idx][1], row)))
	return out


def cost_func(thetas, depend_on, target):
	"""
		thetas:    |
			t_0: 1 |	x_0: 1 1 1
			t_1: 1 |	x_1: 2 3 4
				   + -------------
						y_ : 3 4 5
	"""
	total = 0
	for idx in range(len(target)):
		cur_vec_x = [row[idx] for row in depend_on]
		hyp = sum([t_i * x_i for (t_i, x_i) in zip(thetas, cur_vec_x)])
		part = math.pow(hyp - target[idx], 2)
		total += part
	total /= 2 * len(target)
	return total


def gradient(thetas, depend_on, target):
	"""
			 thetas:   |
				t_0: 1 |	x_0: 1 1 1
				t_1: 1 |	x_1: 2 3 5
					   + -------------
					   |	y_ : 3 4 5
					   + -------------
					   |  diff : 0 0 1
	"""
	grad = []
	diff = []

	for idx in range(len(target)):
		cur_vec_x = [row[idx] for row in depend_on]
		cur_diff = sum([t_i * x_i for (t_i, x_i) in zip(thetas, cur_vec_x)]) - target[idx]
		diff.append(cur_diff)

	for row in depend_on:  # row is a values of x_i for all points
		cur_grad = sum([cur_d * cur_x for (cur_d, cur_x) in zip(diff, row)]) / len(row)
		grad.append(cur_grad)

	return grad


def gradient_descent(thetas, depend_on, target, learn_rate):
	grad = gradient(thetas, depend_on, target)
	for idx in range(len(thetas)):
		thetas[idx] -= learn_rate * grad[idx]
	return thetas, cost_func(thetas, depend_on, target)


def regression(target, depend_on, limit_iter=0, learn_rate=0.5):
	if limit_iter < 1:
		limit_iter = MAX_ITER

	thetas = [0] * len(depend_on)
	window = deque(maxlen=WIN_SIZE)

	loss = cost_func(thetas, depend_on, target)
	window.append(loss)
	yield thetas, loss

	for _ in range(limit_iter):
		thetas, loss = gradient_descent(thetas, depend_on, target, learn_rate)
		window.append(loss)
		yield thetas, loss

		if len(window) == window.maxlen \
		   and max([abs(window[-1] - elem) for elem in window]) < ERR_EPSILON:
			break


def apply_theta(thetas, depend_on):
	hyp = []
	for idx in range(len(depend_on[0])):
		cur_vec_x = [row[idx] for row in depend_on]
		hyp.append(sum([t_i * x_i for (t_i, x_i) in zip(thetas, cur_vec_x)]))
	return hyp


def predict(vector, std, theta):
	vector = standardization_apply(vector, std)
	vector = [1.0, *vector]
	out = 0
	for idx in range(len(theta)):
		out += vector[idx] * theta[idx]
	return out