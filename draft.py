import time
from random import random
import matplotlib.pyplot as plt
import numpy as np


y_time_result = []
x_size = []
values = np.linspace(0,100000, 1000)

for len_array in values:
    current_list = []

    start = time.time()
    for i in range(int(len_array)):
        current_list.append(random())
    end = time.time()

    delay = end - start
    y_time_result.append(delay)
    x_size.append(int(len_array))

    print(f"size {int(len_array)} |delay: {delay:.4f}")

plt.figure()
plt.plot(x_size, y_time_result)
plt.show()

