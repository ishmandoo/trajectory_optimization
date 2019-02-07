import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n = 50
starting_theta = 0
target_thetas = torch.FloatTensor([3.1415] * n)
ts = torch.zeros(n, requires_grad=True)
dt = 0.1


def update_thetas(ts_new):
	thetas = torch.zeros(n)
	omegas = torch.zeros(n)
	thetas[0] = starting_theta
	omegas[0] = 0
	for i in range(1, len(ts_new)):
		old_omega = omegas[i-1].clone()
		old_theta = thetas[i-1].clone()
		omegas[i] = old_omega + (torch.clamp(ts_new[i],-1.,1.) + 0.1 * torch.sin(old_theta)) * dt
		thetas[i] = old_theta + (old_omega * dt)
	return thetas

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
 	