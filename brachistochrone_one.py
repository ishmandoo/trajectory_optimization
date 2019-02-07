import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random

starting_y = 0.25 # y position of the beginning of the wire, end always at 0
xs = torch.linspace(0,1,25) # evenly spaced points in x, from 0 to 1
#ys = torch.linspace(starting_y,starting_y,25, requires_grad=True) # start with a straight line connecting start to end
ys = torch.randn(25).mul(0.1).requires_grad_()
with torch.no_grad():
	ys[0] = starting_y
	ys[-1] = 0.
print(ys)

# the physics
def times(ys): # calculate the time to get to each point
	vs = torch.sqrt(starting_y - ys) # find velocity based on height
	dys = ys[1:] - ys[:-1] # find y difference
	dx = xs[1]-xs[0] # find x  difference
	lengths = torch.sqrt(dx.pow(2) + dys.pow(2)) # calculate arc length
	return torch.cumsum((2./(vs[:-1]+vs[1:])) * lengths, dim=0) # integrate to find time to each point using midpoint velocity


fig = plt.figure()
ax = fig.add_subplot(111)

# solving for the cycloid solution
def func(theta):
	return (1.-np.cos(theta)) / (theta - np.sin(theta)) - .25

theta = optimize.brentq(func, 0.01, 2*np.pi)

# parameterized cycloid solution
r = 1 / (theta - np.sin(theta))
ts = np.linspace(0,2 * np.pi,100)
sol_xs = r * (ts-np.sin(ts))
sol_ys = starting_y - r * (1-np.cos(ts))

ax.plot(sol_xs, sol_ys)

li, = ax.plot(xs.numpy(), ys.detach().numpy())

fig.canvas.draw()
plt.show(block=False)

# a pause
raw_input()

# gradient descent
lr = 0.005 # learning rate
while True:
	ts = times(ys) # get the times to get to each point
	t = ts[-1] # the time to get to the final point
	t.backward() # back propagate to find gradients
	grads = ys.grad # hold on to the gradient
	print(grads)

	# update plots
	li.set_xdata(xs.numpy())
	li.set_ydata(ys.detach().numpy())
	fig.canvas.draw()

	# update wire heights
	with torch.no_grad():
		i = random.choice(range(1,len(grads)-1))
		ys[i] -= lr * grads[i]
	ys.grad.data.zero_() # zero gradients
