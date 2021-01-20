import csv


def load(file_name):
	try:
		with open(file_name, newline='') as csvfile:
			tab = csv.reader(csvfile, delimiter=',')

			labels = next(tab, None)
			if labels is None:
				raise RuntimeError(f"Empty file. Check data file")

			check_row_len = len(labels)
			last_file_line = 2  # start from 0 and labels line
			raw_data = []

			for row in tab:
				last_file_line += 1
				if check_row_len != len(row):
					raise IndentationError(f"Line {last_file_line}: miss value in line. Check data file")
				raw_data.append(tuple(map(float, row)))

			return labels, sorted(raw_data)

	except ValueError as e:
		print(f"Line {last_file_line}: {e}. Check data file")
		quit(1)

	except IndentationError as e:
		print(e)
		quit(1)

	except FileNotFoundError as e:
		print(e)
		quit(1)

	except RuntimeError as e:
		print(e)
		quit(1)

