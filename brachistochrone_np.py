import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(0,1,25)
ys = np.zeros(25)
ys[0] = 1
ys[1] = 0
ys[2] = 0


def speeds(ys):
	return ys[0] - ys

def times(ys):
	vs = speeds(ys)
	slopes = np.diff(ys)/(xs[1]-xs[0])
	slowdown = 1./np.cos(np.arctan(slopes))
	#plt.plot(xs[1:], slowdown)
	return np.append([0], np.cumsum(vs[1:] * slowdown))


plt.plot(xs, ys)
plt.plot(xs, times(ys))
plt.show()
