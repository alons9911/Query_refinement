import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 15})


x_list = list()
x_naive = list()
execution_timeps = list()
execution_timelt = list()

input_path = r'time_0.csv'
input_file = open(input_path, "r")

Lines = input_file.readlines()

count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if line == '\n':
        continue
    if count < 2:
        continue
    if count > 8:
        break
    items = line.strip().split(',')
    x_list.append(items[0])
    execution_timeps.append(float(items[1]))
    # execution_timelt.append(float(items[2]))

print(x_list, execution_timeps, execution_timelt)





n_groups = len(x_list)

# create plot
fig, ax = plt.subplots(figsize=(5, 4))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.7


rects1 = plt.bar(index, execution_timeps, bar_width,
                 alpha=opacity,
                 color='blue',
                 label='PS')

# rects2 = plt.bar(index + bar_width, execution_timelt, bar_width,
#                  alpha=opacity,
#                  color='green',
#                  label='LT')

plt.xlabel('Query and Constraint')
plt.ylabel('Running time (s)')
plt.xticks(index + bar_width, x_list)
plt.yticks([0, 10, 20, 30])
plt.legend()

plt.tight_layout()
plt.savefig("adult_time_0.png",
            bbox_inches='tight')
plt.show()
