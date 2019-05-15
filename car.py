import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n_steps = 4 # number of steps to test the var performance
#speeds = torch.tensor(np.random.normal(1., .1, size=n_steps).astype(np.float32), requires_grad=True) # init random wheel point radii
#steers = torch.tensor(np.random.normal(1., .1, size=n_steps).astype(np.float32), requires_grad=True) # init random wheel point radii
speeds = torch.ones(n_steps, requires_grad=True) # init random wheel point radii
steers = torch.ones(n_steps, requires_grad=True) # init random wheel point radii
dt = .1 # time step for dynamics

target_pos = torch.tensor([1.,4.])

# finds the trajectory from the speed and steering commands
def test_car(speeds, steers):
    angles = torch.zeros(n_steps) #[torch.zeros(1)] * n_steps # torch.zeros(n_steps)
    poss = torch.zeros(n_steps, 2) #[[torch.zeros(1),torch.zeros(1)]] * n_steps # 

    # simulate the car performance
    for i in range(1,n_steps):
        speed =  speeds[i-1] #torch.clamp(speeds[i-1], -1., 1.)
        steer = steers[i-1] #torch.clamp(steers[i-1], -1., 1.)
        #temp = torch.zeros(2)
        #temp[0] = torch.cos(angles[i-1])
        #temp[1] = torch.sin(angles[i-1])
        poss[i][0] += poss[i-1][0] + dt * speed * torch.cos(angles[i-1])
        poss[i][1] += poss[i-1][1] + dt * speed * torch.sin(angles[i-1])
        #poss[i] = poss[i-1] + dt * speed * torch.tensor([0.5,0.5])# torch.tensor([torch.cos(angles[i-1]), torch.sin(angles[i-1])], requires_grad=False)
        angles[i] += angles[i-1] + dt * speed * steer
    return poss, angles

'''
# list to keep track of objective progress
final_poss = []

# objective plot
fig_obj = plt.figure()
ax_obj = fig_obj.add_subplot(111)
ax_obj.autoscale(enable=True, axis="y", tight=False)

li_obj, = ax_obj.plot(final_poss)


fig_in = plt.figure()
ax_in = fig_in.add_subplot(111)
ax_in.autoscale(enable=True, axis="y", tight=False)

li_in, = ax_obj.plot(speeds.detach().numpy())
'''

# radii plot
fig_traj = plt.figure()
ax_traj = fig_traj.add_subplot(111)

li_traj, = ax_traj.plot([])


fig_traj.canvas.draw()
plt.show(block=False)
optimizer =  torch.optim.SGD([speeds, steers], lr=0.1)
#optimizer = torch.optim.Adam([radii])

while True:
    poss, angles = test_car(speeds, steers)
    #print(poss, angles)
    #final_poss.append(poss[-1] - target_pos)
    print("WHAT", poss - target_pos)
    cost = torch.sum(torch.norm(poss- target_pos, dim=1))
    print(cost)
    optimizer.zero_grad()
    cost.backward() #retain_graph=True)
    #torch.autograd.grad(cost, [speeds])
    #torch.autograd.grad(cost, [steers])
    print(speeds.grad, steers.grad)
    print(speeds)
    '''
    # update plots
    li_obj.set_data(range(len(final_poss)),final_poss)
    li_in.set_ydata(speeds.detach().numpy())
    '''
    #ax_obj.relim()
    #ax_obj.autoscale_view()

    ax_traj.clear()
    ax_traj.set_aspect('equal', 'datalim')
    xs, ys = poss.detach().numpy().transpose()
    ax_traj.plot(xs, ys)

    fig_traj.canvas.draw()
    #fig_obj.canvas.draw()
    #fig_in.canvas.draw()
    


    # update radii
    optimizer.step()
     
