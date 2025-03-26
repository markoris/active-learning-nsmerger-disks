import sys

idxs = [3, 1, 5, 2, 1]

string = sys.argv[1]
string = string.split('_')
for idx in range(len(string)):
	string[idx] = string[idx][idxs[idx]:]
	print(float(string[idx]))
#	string[idx] = float(string[idx])
