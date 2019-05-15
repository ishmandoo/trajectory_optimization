import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n = 10 # number of wheel points
n_steps = 1000 # number of steps to test the wheel performance
#radii = torch.tensor([2.] + [1.] * (n-1), requires_grad=True) # init wheel radii with one bump
radii = torch.tensor(np.random.normal(1., .1, size=n).astype(np.float32), requires_grad=True) # init random wheel point radii
dtheta = 2. * np.pi / n # angle between wheel points
thetas = torch.arange(0,2*np.pi, dtheta) # list of wheel point angles
dt = 0.1 # time step for dynamics
torque = 1. # torque on the wheel
mass = 10. # mass of the car

# finds the final speed of the wheel from radii
def test_wheel(radii):
    # normalize the radii so that changing the size has no effect
    norm_radii = radii / torch.mean(radii)

    # find the coordinates of the wheel points and hub
    edge_positions, hub_position = wheel_coordinates(norm_radii)

    speeds = torch.zeros(n_steps) # speed of the cart after each step

    # simulate the wheel performance
    for i in range(1,n_steps):
        # find pivot properties
        pivot_index = torch.argmin(edge_positions[:,1])
        pivot_radius = norm_radii[pivot_index]
        pivot_vector = edge_positions[pivot_index] - hub_position
        pivot_angle = torch.atan2(pivot_vector[1], pivot_vector[0])

        # calculate force at the pivot point
        force = force_compoment * torque / pivot_radius

        # calculate the component of pivot force that pushes the car forward
        force_compoment = torch.cos(pivot_angle + np.pi/2)

        # calculate the new speed of the car and the rotational speed of the wheel
        speeds[i] = speeds[i-1] + (force * dt / mass)
        rotational_speed = speeds[i] / pivot_radius

        
        dangle = rotational_speed * dt
        
        # rotate the wheel
        edge_positions, hub_position = rotate(edge_positions, hub_position, dangle)

    return speeds

def draw_wheel(edge_positions):
    xs, ys = edge_positions.transpose(0,1).detach().numpy()
    plt.scatter(xs, ys)
    
def rotation_matrix(theta):
    return torch.tensor([[torch.cos(theta), torch.sin(theta)],[-torch.sin(theta), torch.cos(theta)]])

def wheel_coordinates(radii):
    edge_positions = (radii.unsqueeze(1) * torch.cat((torch.cos(thetas.reshape((-1,1))),torch.sin(thetas.reshape((-1,1)))), 1))
    hub_position = torch.tensor([0.,0.])
    return edge_positions, hub_position


def rotate(edge_positions, hub_position, theta):
    rot_matrix = rotation_matrix(theta)
    edge_positions = torch.matmul(rot_matrix, edge_positions.transpose(0,1)).transpose(0,1)
    hub_position = torch.matmul(rot_matrix,hub_position)

    pivot_index = torch.argmin(edge_positions[:,1])
    hub_position -= torch.tensor(edge_positions[pivot_index])
    edge_positions -= torch.tensor(edge_positions[pivot_index])
    return edge_positions, hub_position

# list to keep track of objective progress
final_speeds = []

# objective plot
fig_obj = plt.figure()
ax_obj = fig_obj.add_subplot(111)
ax_obj.autoscale(enable=True, axis="y", tight=False)

li_obj, = ax_obj.plot(final_speeds)

# radii plot
fig_rad = plt.figure()
ax_rad = fig_rad.add_subplot(111)

li_rad, = ax_rad.plot(radii.detach().numpy())

# wheel shape plot
fig_wheel = plt.figure()
ax_wheel = fig_wheel.add_subplot(111)
ax_wheel.set_aspect('equal', 'datalim')

edge_positions, hub_position = wheel_coordinates(radii)
edge_xs, edge_ys = edge_positions.transpose(0,1).detach().numpy()
hub_x, hub_y = hub_position
li_wheel, = ax_wheel.fill(edge_xs, edge_ys)
li_hub, = ax_wheel.plot(hub_x, hub_y, 'b.')

fig_obj.canvas.draw()
fig_rad.canvas.draw()
fig_wheel.canvas.draw()
plt.show(block=False)
optimizer =  torch.optim.SGD([radii], lr=0.01)
#optimizer = torch.optim.Adam([radii])

while True:
    speeds = test_wheel(radii)
    final_speeds.append(speeds[-1])
    cost = -speeds[-1]
    optimizer.zero_grad()
    cost.backward(retain_graph=False)

    # update plots
    li_obj.set_data(range(len(final_speeds)),final_speeds)
    li_rad.set_ydata(radii.detach().numpy())

    ax_obj.relim()
    ax_obj.autoscale_view()


    edge_positions, hub_position = wheel_coordinates(radii)
    edge_xs, edge_ys = edge_positions.transpose(0,1).detach().numpy()
    hub_x, hub_y = hub_position
    ax_wheel.clear()
    ax_wheel.set_aspect('equal', 'datalim')
    ax_wheel.fill(edge_xs, edge_ys)
    ax_wheel.plot(hub_x, hub_y, 'b.')

    fig_obj.canvas.draw()
    fig_rad.canvas.draw()
    fig_wheel.canvas.draw()


    # update radii
    optimizer.step()
     
