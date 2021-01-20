import argparse
import json
import ml

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.txt')
parser.add_argument('vector', nargs='+', type=float)
args = parser.parse_args()

try:
	with open(args.config) as json_file:
		param = json.load(json_file)

	if len(param['std']) != len(args.vector):
		raise ValueError(f'Incorrect vector size. Length of vector should be {len(param["std"])}')

	print(ml.predict(args.vector, param['std'], param['theta']))

except ValueError as e:
	print(e)
	quit(1)

except FileNotFoundError as e:
	print(e)
	quit(1)

