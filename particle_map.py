import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n = 200
pos_target = torch.FloatTensor([10.,10.])
vs = torch.FloatTensor([[0.1,0.11 ] ]* n)
vs.requires_grad_()
dt = 0.1


def update_poss(test_vs):
	poss = torch.zeros([n,2])
	for i in range(1, n):
		old_pos = poss[i-1,:].clone()
		v = test_vs[i-1,:].clone()
		#poss[i,:] = old_pos + dt * (v/torch.norm(v)) * (old_pos[0]+1)
		#print(int(not (old_pos[0] < 6 and old_pos[0]  >4 and old_pos[1] < 6 and old_pos[1] > 4)))

		if (torch.norm(v) > 1.):
			v = (v.clone()/torch.norm(v.clone()))
		poss[i,:] = old_pos + dt * v  * float(not (old_pos[0] < 6 and old_pos[0]  >4 and old_pos[1] < 7 and old_pos[1] > 5))
	return poss


poss = update_poss(vs)

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(131)
ax1.autoscale(enable=True, axis="xy")

li1, = ax1.plot(poss[:,0].detach().numpy(),poss[:,1].detach().numpy(),'.')

ax2 = fig.add_subplot(132)
ax2.autoscale(enable=True)

li2, = ax2.plot(vs[:-1,0].detach().numpy())


ax3 = fig.add_subplot(133)
ax3.autoscale(enable=True)
li3, = ax3.plot(vs[:-1,1].detach().numpy())

fig.canvas.draw()
plt.show(block=False)
#optimizer =  torch.optim.SGD([ts], lr=0.0000000001, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam([vs])

while True:
	poss = update_poss(vs)
	cost = torch.norm(poss[-1,:] - pos_target)
	optimizer.zero_grad()
	cost.backward(retain_graph=False)

	# update plots
	li1.set_xdata(poss.detach().numpy()[:,0])
	li1.set_ydata(poss.detach().numpy()[:,1])
	li2.set_ydata(vs.detach().numpy()[:-1,0])
	li3.set_ydata(vs.detach().numpy()[:-1,1])
	ax1.relim()
	ax2.relim()
	ax3.relim()
	# update ax.viewLim using the new dataLim
	ax1.autoscale_view()
	ax2.autoscale_view()
	ax3.autoscale_view()

	fig.canvas.draw()


	# update wire heights
	optimizer.step()
 	