import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n = 200
xs = [0] * n
target_xs = torch.FloatTensor([1] * 100 + [0] * (n-100))
vs = [0] * n
fs = torch.linspace(0,0,n, requires_grad=True)


def update_xs(fs):
	vs_new = torch.cumsum(fs, dim=0)
	xs_new = torch.cumsum(vs_new, dim=0)
	return xs_new, vs_new

fig = plt.figure()
plt.autoscale(tight=True)
ax = fig.add_subplot(111)
ax.autoscale(enable=True, axis="y", tight=False)


li1, = ax.plot(xs)
li2, = ax.plot(vs)
li3, = ax.plot(fs.detach().numpy())

fig.canvas.draw()
plt.show(block=False)

#optimizer =  torch.optim.SGD([fs], lr=0.000000005, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam([fs])



lr = 0.000001 # learning rate
while True:
	xs, vs = update_xs(fs)
	#print(xs, vs)
	cost = torch.sum((xs - target_xs).pow(2) + fs.pow(2))
	optimizer.zero_grad()
 	cost.backward()
 

	# update plots
	li1.set_ydata(xs.detach().numpy())
	li2.set_ydata(vs.detach().numpy())
	li3.set_ydata(fs.detach().numpy())
	ax.relim()
	# update ax.viewLim using the new dataLim
	ax.autoscale_view()

	fig.canvas.draw()

	# update wire heights
 	optimizer.step()