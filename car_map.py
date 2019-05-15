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

def box(x, y):
    #return 1.0
    return float(not ((1.<x<2.) and (1.<y<2.))) + 0.01

def wall(x):
    return float(not (x>3.5)) + 0.01

# finds the trajectory from the speed and steering commands
def test_car(speeds, steers):
    angles =[torch.tensor(0.)]
    xs = [torch.tensor(0.)]
    ys = [torch.tensor(0.)]

    # simulate the car performance
    for i in range(1,n_steps):
        speed =  torch.clamp(speeds[i-1], -1., 1.) # limit the speed
        steer = torch.clamp(steers[i-1], -10., 10.) # limit the steering angle
        #terrain = wall(xs[-1]) # calculate terrain modifier
        #terrain = box(xs[-1], ys[-1])
        terrain = 1.
        xs.append(xs[-1]+ dt * terrain * speed * torch.cos(angles[-1]))
        ys.append(ys[-1]+ dt * terrain * speed * torch.sin(angles[-1]))
        angles.append((angles[i-1] + dt * speed * steer)%(2*np.pi))
    return xs, ys, angles


# list to keep track of objective progress
costs = []
angle_costs = []
dist_costs = []

# objective plot
fig_obj = plt.figure()
ax_obj = fig_obj.add_subplot(111)
ax_obj.autoscale(enable=True, axis="y", tight=False)
ax_obj.set_xlabel("iteration")
ax_obj.set_ylabel("cost")

li_obj, = ax_obj.plot([],[])
li_obj_ang, = ax_obj.plot([],[])
li_obj_dist, = ax_obj.plot([],[])


fig_in = plt.figure()
ax_in = fig_in.add_subplot(111)
ax_in.autoscale(enable=True, axis="y", tight=False)

li_sp, = ax_in.plot(speeds.detach().numpy())
li_st, = ax_in.plot(steers.detach().numpy())

# radii plot
fig_traj = plt.figure()
ax_traj = fig_traj.add_subplot(111)

li_traj, = ax_traj.plot([])


fig_obj.canvas.draw()
fig_traj.canvas.draw()
fig_in.canvas.draw()
plt.show(block=False)
optimizer =  torch.optim.SGD([speeds, steers], lr=0.001, momentum=.01, nesterov=True)

#optimizer =  torch.optim.Adam([speeds, steers])

for i in range(1000):
    xs, ys, angles = test_car(speeds, steers)

    angle_error = target_angle - angles[-1]
    angle_cost = n_steps * torch.min(torch.min((angle_error+2*np.pi)**2, (angle_error-2*np.pi)**2), angle_error**2)

    dist_cost = sum([(target_pos[0] - xs[i])**2 + (target_pos[1] - ys[i])**2 for i in range(len(xs))])    
    #cost = (target_pos[0] - xs[-1])**2 + (target_pos[1] - ys[-1])**2
    cost = dist_cost + angle_cost

    costs.append(cost)
    angle_costs.append(angle_cost)
    dist_costs.append(dist_cost)
    optimizer.zero_grad()
    cost.backward() #retain_graph=True)
    
    # update plots
    li_obj.set_data(range(len(costs)),costs)
    li_obj_ang.set_data(range(len(angle_costs)),angle_costs)
    li_obj_dist.set_data(range(len(dist_costs)),dist_costs)
    li_sp.set_ydata(speeds.detach().numpy())
    li_st.set_ydata(steers.detach().numpy())
    
    ax_obj.relim()
    ax_obj.autoscale_view()
    ax_obj.legend(["total", "angle", "dist"])

    ax_in.relim()
    ax_in.autoscale_view()
    ax_in.legend(["speed", "steer"])

    ax_traj.clear()
    ax_traj.set_aspect('equal', 'datalim')
    ax_traj.plot(xs, ys)

    fig_traj.canvas.draw()
    fig_obj.canvas.draw()
    fig_in.canvas.draw()
    


    # update radii
    optimizer.step()


    fig_traj.savefig("vid_1/path_{i}.jpg".format(i=i))
    fig_obj.savefig("vid_1/obj.jpg")
     
