import os
import csv
import math
import argparse
import textwrap
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

# print(raw_data)
raw_data.sort()
# print()
# print(raw_data)

data = None
for pack in raw_data:
	if data is None:
		data = list()
		for _ in row:
			data.append(list())
	for idx in range(len(pack)):
		data[idx].append(pack[idx])

# print(labels)
# todo add sorting by selected x_i

# print(data)
result = data[-1]
depend_on = data[0:-1]


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


def standardization(matrix):
	tmp_means = means(matrix)
	tmp_sigmas = sigmas(matrix, tmp_means)
	out = []
	for idx, row in enumerate(matrix):
		out.append(list(map(lambda x: (x - tmp_means[idx]) / tmp_sigmas[idx], row)))
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


def create_test_files():
	test_file = 'test_data_0.csv'
	if not os.path.exists(test_file):
		with open(test_file, 'w') as file:
			file.write("km,price\n240000,3650\n139800,3800\n150500,4400\n"
					   "185530,4450\n176000,5250\n114800,5350\n166800,5800\n"
					   "89000,5990\n144500,5999\n84000,6200\n82029,6390\n"
					   "63060,6390\n74000,6600\n97500,6800\n67000,6800\n"
					   "76025,6900\n48235,6900\n93000,6990\n60949,7490\n"
					   "65674,7555\n54000,7990\n68500,7990\n22899,7990\n"
					   "61789,8290\n")
		print(f"test file 1/2 - {test_file} - created")
	else:
		print(f"test file 1/2 - {test_file} - Failed to create. File already exists")

	test_file = 'test_data_1.csv'
	if not os.path.exists(test_file):
		with open(test_file, 'w') as file:
			file.write("km,km2,price\n240000,57600000000,3650\n"
					   "139800,19544040000,3800\n150500,22650250000,4400\n"
					   "185530,34421380900,4450\n176000,30976000000,5250\n"
					   "114800,13179040000,5350\n166800,27822240000,5800\n"
					   "89000,7921000000,5990\n144500,20880250000,5999\n"
					   "84000,7056000000,6200\n82029,6728756841,6390\n"
					   "63060,3976563600,6390\n74000,5476000000,6600\n"
					   "97500,9506250000,6800\n67000,4489000000,6800\n"
					   "76025,5779800625,6900\n48235,2326615225,6900\n"
					   "93000,8649000000,6990\n60949,3714780601,7490\n"
					   "65674,4313074276,7555\n54000,2916000000,7990\n"
					   "68500,4692250000,7990\n22899,524364201,7990\n"
					   "61789,3817880521,8290\n")
		print(f"test file 2/2 - {test_file} - created")
	else:
		print(f"test file 2/2 - {test_file} - Failed to create. File already exists")


class TestAsHelp(argparse.Action):

	def __call__(self, *args, **kwargs):
		create_test_files()
		parser.exit()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		formatter_class=argparse.RawDescriptionHelpFormatter,
		prog='lin_reg_train',
		epilog=textwrap.dedent(
			'\n'
			'Remark: program wait .csv file with next structure:\n'
			'	+ ------------- first variable\n'
			'	|  + ---------- more variables separated by ","\n'
			'	|  |   + ------ target value\n'
			'	|  |   |\n'
			'\n'
			'	km,...,price  - first line: labels\n'
			'	21,...,500    - second line: first point\n'
			'	53,...,496    - third line: second point\n'
			'	.\n'
			'	.\n'
			'	.\n'
		))

	parser.add_argument('-t', '--test', nargs=0, action=TestAsHelp,
						help='create two .csv examples files')
	parser.add_argument('path_to_file', type=str)

	args = parser.parse_args()



# depend_on = standardization(depend_on)
# regression(result, depend_on)
