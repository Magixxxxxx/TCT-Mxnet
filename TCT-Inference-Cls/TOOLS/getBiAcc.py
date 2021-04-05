import sys

input_file = sys[1]
gt = int(sys[2]) #正确标签

right = 0
false = 0

with open(input_file) as f:
	for row in f:
		p = int(row.strip('\n').rpartition(' ')[-1])
		if abs(p-gt)<0.5 right+=1 else false+=1
total = right+false

print('right:{} percent:{}'.format(right, right/total))
print('false:{} percent:{}'.format(false, false/total))
