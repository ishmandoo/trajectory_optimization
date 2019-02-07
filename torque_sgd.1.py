import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n = 50
starting_theta = 0.1
target_thetas = torch.FloatTensor([0] * n)
ts = torch.zeros(n, requires_grad=True)


def update_thetas(ts_new):
	thetas = np.zeros(n)
	thetas[0] = starting_theta
	theta = starting_theta
	w = 0.
	for i in range(1, len(ts_new)):
		theta += w / 2.
		thetas[i] = theta
		w += ts_new[i] + np.sin(thetas[i])
		theta += w / 2.
		thetas[i] = theta
	return torch.FloatTensor(thetas)

fig = plt.figure()
plt.autoscale(tight=True)
ax = fig.add_subplot(111)
ax.autoscale(enable=True, axis="y", tight=False)

thetas = update_thetas(ts)

li1, = ax.plot(thetas.detach().numpy())
li3, = ax.plot(ts.detach().numpy())

fig.canvas.draw()
plt.show(block=False)
#optimizer =  torch.optim.SGD([ts], lr=0.0000000001, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam([ts])

while True:
	thetas = update_thetas(ts)
	#print(thetas, ws)
	cost = torch.sum((thetas - target_thetas).pow(2))
	optimizer.zero_grad()
 	cost.backward(retain_graph=False)
 

	# update plots
	li1.set_ydata(thetas.detach().numpy())
	li3.set_ydata(ts.detach().numpy())
	ax.relim()
	# update ax.viewLim using the new dataLim
	ax.autoscale_view()

	fig.canvas.draw()

	# update wire heights
 	optimizer.step()
 	