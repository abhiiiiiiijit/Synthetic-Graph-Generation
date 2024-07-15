import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Create a simple plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Simple Plot')
plt.show()

plt.savefig('plot.png')
