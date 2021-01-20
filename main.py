import parser_n_flags
import check_file
import ml
import plotting
import json


args = parser_n_flags.arg_parser()
labels, data = check_file.load(args.path_to_file)
t_data = ml.transpose_data(data)

# split variables and target value
result = t_data[-1]
depend_on = t_data[0:-1]

# standardization
depend_std = ml.standardization(depend_on)

# add row for theta_0
for_theta_zero = [1.0] * len(result)
depend_std = [for_theta_zero, *depend_std]

# calc
loss_mem = []
theta = None
rate = args.rate
limi = args.limi

for cur_theta, loss in ml.regression(result, depend_std, limit_iter=limi, learn_rate=rate):
	loss_mem.append(loss)
	theta = cur_theta
	if args.iter:
		plotting.show_graphs(depend_on[0], theta, depend_std, result, loss_mem)
if args.show:
	plotting.show_graphs(depend_on[0], theta, depend_std, result, loss_mem)


outdata = {
	"std": ml.standardization_vector(depend_on),
	"theta": theta
}
with open('config.txt', 'w') as outfile:
	json.dump(outdata, outfile, indent=2)

# [print(elem) for elem in zip(depend_on[0], ml.apply_theta(theta, depend_std))]
