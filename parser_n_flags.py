import os
import argparse

help_txt = (
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
)

test_file_one = (
	"km,price\n240000,3650\n139800,3800\n150500,4400\n"
	"185530,4450\n176000,5250\n114800,5350\n166800,5800\n"
	"89000,5990\n144500,5999\n84000,6200\n82029,6390\n"
	"63060,6390\n74000,6600\n97500,6800\n67000,6800\n"
	"76025,6900\n48235,6900\n93000,6990\n60949,7490\n"
	"65674,7555\n54000,7990\n68500,7990\n22899,7990\n"
	"61789,8290\n"
)

test_file_two = (
	"km,km2,price\n240000,57600000000,3650\n"
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
	"61789,3817880521,8290\n"
)


def create_test_files():
	test_set = [
		('test_data_1.csv', test_file_one),
		('test_data_2.csv', test_file_two)
	]

	for idx, pack in enumerate(test_set):
		idx += 1
		elem_name, elem_data = pack

		if not os.path.exists(elem_name):
			with open(elem_name, 'w') as file:
				file.write(elem_data)
			print(f"test file {idx}/{len(test_set)} - {elem_name} - created")
		else:
			print(
				f"test file {idx}/{len(test_set)}  - {elem_name} - Failed "
				f"to create. File already exists"
			)


class TestAsHelp(argparse.Action):

	def __call__(self, parser, namespace, values, option_string=None):
		create_test_files()
		parser.exit()


def arg_parser():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.RawDescriptionHelpFormatter,
		prog='lin_reg_train',
		epilog=help_txt
	)
	parser.add_argument('path_to_file', type=str)

	parser.add_argument('-t', '--test', nargs=0, action=TestAsHelp,
						help='create two .csv examples files')
	parser.add_argument('-s', '--show', action='store_true')
	parser.add_argument('-i', '--iter', action='store_true')
	parser.add_argument('-r', '--rate', type=float, default=0.5)
	parser.add_argument('-l', '--limi', type=int, default=0)

	return parser.parse_args()
