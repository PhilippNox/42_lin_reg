import csv
import math
import matplotlib.pyplot as plt

with open('data_poly.csv', newline='') as csvfile:
	tab = csv.reader(csvfile, delimiter=',')

	labels = None
	raw_data = []

	for idx, row in enumerate(tab):
		if idx == 0 and any([not elem.isdigit() for elem in row]):
			labels = row
			continue
		raw_data.append(tuple(map(float, row)))

print(raw_data)
raw_data.sort()
print()
print(raw_data)

data = None
for pack in raw_data:
	if data is None:
		data = list()
		for _ in row:
			data.append(list())
	for idx in range(len(pack)):
		data[idx].append(pack[idx])


#print(labels)
# todo add sorting by selected x_i

print(data)
result = data[-1]
depend_on = data[0:-1]


def means(matrix):
	out = []
	for row in matrix:
		out.append(sum(row)/len(row))
	return out


def sigmas(matrix, means=None):
	out = []
	for idx, row in enumerate(matrix):
		tmp_mean = sum(row)/len(row) if means is None else means[idx]
		tmp_units = map(lambda x: math.pow(x - tmp_mean, 2), row)
		out.append(math.sqrt(sum(tmp_units) / len(row)))
	return out


def standardization(matrix):
	tmp_means = means(matrix)
	tmp_sigmas = sigmas(matrix, tmp_means)
	out = []
	for idx, row in enumerate(matrix):
		out.append(list(map(lambda x: (x - tmp_means[idx])/tmp_sigmas[idx], row)))
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
		hyp = sum([t_i*x_i for (t_i, x_i) in zip(thetas, cur_vec_x)])
		part = math.pow(hyp - target[idx], 2)
		total += part
	total /= 2*len(target)
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
		cur_grad = sum([cur_d * cur_x for (cur_d, cur_x) in zip(diff, row)])/len(row)
		grad.append(cur_grad)

	return grad


def hyp_target(thetas, depend_on, target):
	hyp = []
	for idx in range(len(target)):
		cur_vec_x = [row[idx] for row in depend_on]
		hyp.append(sum([t_i * x_i for (t_i, x_i) in zip(thetas, cur_vec_x)]))
	return hyp


def regression(target, depend_on, limit_iter=10, learn_rate=0.5):
	# add row for theta_0
	for_theta_zero = [1.0] * len(target)
	depend_on = [for_theta_zero, *depend_on]

	# setup thetas
	thetas = [0] * len(depend_on)

	plt.figure()
	n_iter = 0
	while n_iter < limit_iter:
		n_iter += 1
		predict = hyp_target(thetas, depend_on, target)
		plt.scatter(depend_on[1], target)
		plt.plot(depend_on[1], predict, color="red", linewidth=2)
		plt.show()

		# cost
		loss = cost_func(thetas, depend_on, target)
		print('theta', thetas)
		print('cost_func', loss, '\n')
		# gradient_descent
		grad = gradient(thetas, depend_on, target)
		for idx in range(len(thetas)):
			thetas[idx] -= learn_rate * grad[idx]


depend_on = standardization(depend_on)
regression(result, depend_on)
