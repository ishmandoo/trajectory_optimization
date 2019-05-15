import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n_steps = 200 # number of steps to test the car performance
speeds = torch.zeros(n_steps-1, requires_grad=True) # init speed control tensor
steers = torch.zeros(n_steps-1, requires_grad=True) # init steering control tensir
dt = .1 # time step for dynamics

target_pos = torch.tensor([3.,3.]) # target position for the car
target_angle = torch.tensor(3*np.pi/2.) # target angle for the car

# finds the trajectory from the speed and steering commands
def test_car(speeds, steers):
    angles =[torch.tensor(0.)]
    xs = [torch.tensor(0.)]
    ys = [torch.tensor(0.)]

    # simulate the car performance
    for i in range(1,n_steps):
        speed =  torch.clamp(speeds[i-1], -1., 1.) # limit the speed
        steer = torch.clamp(steers[i-1], -10., 10.) # limit the steering angle
        xs.append(xs[-1]+ dt * speed * torch.cos(angles[-1]))
        ys.append(ys[-1]+ dt * speed * torch.sin(angles[-1]))
        angles.append((angles[i-1] + dt * speed * steer)%(2*np.pi))
    return xs, ys, angles


# list to keep track of objective progress
costs = []
angle_costs = []
dist_costs = []

# cost plot
fig_cost = plt.figure()
ax_cost = fig_cost.add_subplot(111)
ax_cost.autoscale(enable=True, axis="y", tight=False)
ax_cost.set_xlabel("iteration")
ax_cost.set_ylabel("cost")

li_cost, = ax_cost.plot([],[])
li_cost_ang, = ax_cost.plot([],[])
li_cost_dist, = ax_cost.plot([],[])

# input plot
fig_in = plt.figure()
ax_in = fig_in.add_subplot(111)
ax_in.autoscale(enable=True, axis="y", tight=False)

li_speed, = ax_in.plot(speeds.detach().numpy())
li_steer, = ax_in.plot(steers.detach().numpy())

# trajectory plot
fig_traj = plt.figure()
ax_traj = fig_traj.add_subplot(111)

li_traj, = ax_traj.plot([])

fig_cost.canvas.draw()
fig_traj.canvas.draw()
fig_in.canvas.draw()
plt.show(block=False)

optimizer =  torch.optim.SGD([speeds, steers], lr=0.001, momentum=.01, nesterov=True)
#optimizer =  torch.optim.Adam([speeds, steers])

for i in range(108):
    # run simulation
    xs, ys, angles = test_car(speeds, steers)

    # calculate costs
    angle_error = target_angle - angles[-1]
    angle_cost = n_steps * torch.min(torch.min((angle_error+2*np.pi)**2, (angle_error-2*np.pi)**2), angle_error**2)

    dist_cost = sum([(target_pos[0] - xs[i])**2 + (target_pos[1] - ys[i])**2 for i in range(len(xs))])    

    cost = dist_cost + angle_cost

    # save costs
    costs.append(cost)
    angle_costs.append(angle_cost)
    dist_costs.append(dist_cost)

    #calculate gradients
    optimizer.zero_grad()
    cost.backward()

    # update inputs
    optimizer.step()

    # update plots
    li_cost.set_data(range(len(costs)),costs)
    li_cost_ang.set_data(range(len(angle_costs)),angle_costs)
    li_cost_dist.set_data(range(len(dist_costs)),dist_costs)
    li_speed.set_ydata(speeds.detach().numpy())
    li_steer.set_ydata(steers.detach().numpy())
    
    ax_cost.relim()
    ax_cost.autoscale_view()
    ax_cost.legend(["total", "angle", "dist"])

    ax_in.relim()
    ax_in.autoscale_view()
    ax_in.legend(["speed", "steer"])

    ax_traj.clear()
    ax_traj.set_aspect('equal', 'datalim')
    ax_traj.plot(xs, ys)

    fig_traj.canvas.draw()
    fig_cost.canvas.draw()
    fig_in.canvas.draw()
    
    fig_traj.savefig("car_vid_1/imgs/path_{i}.jpg".format(i=i))
    fig_cost.savefig("car_vid_1/obj.jpg")
    fig_in.savefig("car_vid_1/in.jpg")
